""" Class definition and functions providing geospatial capabilitis for planet Earth. (alias: Tiling API)
Functionality:
- Create tiles based on H3 (Uber API) resolution
- Utility functionality to conveniently find tiles at certain resolutions, landmass tiles, and tiles within a circular region.
- Supports fine-grained tiling of the Earth's surface and rendering of 2D and 3D shapes with individual tile data properties.
- Efficiently encode tile data in binary format for transfer to the frontend.


Author: Michael Petry
Last Edit: 22.08.2024

Documentation is in Google Docstrings format.
"""

import numpy as np
from typing import Optional, List, Dict
from pylib.InheritanceUtils import RequireEncode
from pylib.Messages import Message
from pylib.Stationary import EarthSurfacePosition
import struct
import base64
import h3
import math
from pylib.Utils import *
    
class Tile():
    r"""Class representing a tile on the Earth's surface.
    
    Args:
        h3index (str): H3 index of the tile in official string format.
        color (Optional[np.int32], optional): Color of the tile encoded as a 32-bit integer in rgba format. Defaults to None.

    Note:

    Additional data properties, such as a height value and volumne for 3D rendering, and a specific 3D color can be set using the setData() method.
    By default, no data property is set and the tile is rendered in 2D only.
    """
    def __init__(self, h3index: str, color: Optional[np.int32]=None):
        self._h3index = h3index
        self._h3indexInt64 = int(h3index, 16)
        self._color = color if color != None else TilingAPI.TileColor.fromRGB(r=255)
        self._data = np.float32(0)
        self._dataColor = color if color != None else TilingAPI.TileColor.fromRGB(r=255)
        self._dataVolume = np.uint8(0)
        self._updateSelf = True
        self._positionlatlon = h3.h3_to_geo(self.getH3Index())
        self._positionsphericalunrotated = latlon_to_spherical(self._positionlatlon[0], self._positionlatlon[1])

    def getH3Index(self, ) -> str:
        r"""Gets the H3 index of the tile.
        
        Returns:
            str: H3 index string of the tile.
        """
        return self._h3index
    
    def setColor(self, color: np.int32):
        r"""Set the color of the tile.
        
        Args:
            color (np.int32): Color of the tile encoded as a 32-bit integer in rgba format.
        """
        if self._color != color:
            self._color = color
            self._update()
    
    def getColor(self,) -> np.int32:
        r"""Get the color of the tile.
        
        Returns:
            np.int32: Color of the tile encoded as a 32-bit integer in rgba format.
        """
        return self._color
    
    def setData(self, height_km: np.float32, color: Optional[np.int32]=None, volume: Optional[np.uint8]=np.uint8(0)):
        r"""Provide extended data properties of the tile. 
         
        Args:
            height_km (np.float32): Height in km for 3D rendering.
            color (Optional[np.int32], optional): Color of the tile encoded as a 32-bit integer in rgba format. If None supplied, color is not overwritten.
            volume (Optional[np.uint8], optional): Volume (thickness) of 3D shape. Defaults to np.uint8(0).
        """
        if (self._data != height_km or self._dataColor != color or self._dataVolume != volume):
            self._data = height_km
            self._dataColor = color if color != None else self.getColor()
            self._dataVolume = volume
            self._update()

    def getData(self,) -> np.float32:
        r"""Get the height of the tile in km.
        
        Returns:
            np.float32: Height of the tile in km.
        """
        return self._data
    
    def getDataColor(self,) -> np.int32:
        r"""Get the data color of the tile used in 3D rendering.
        
        Returns:
            np.int32: Color of the tile encoded as a 32-bit integer in rgba format.
        """
        return self._dataColor
    
    def getDataVolume(self,) -> np.uint8:
        r"""Get the volume (thickness) of the 3D shape of the tile.
        
        Returns:
            np.uint8: Volume (thickness) of the 3D shape. 0 means zero volume, 255 means 100% volume."""
        return self._dataVolume
    
    def getPosition(self,) -> List[float]:
        r"""Get the position of the tile in latitude and longitude.
        
        Returns:
            List[float]: Position of the tile in latitude and longitude.
        """
        return self._positionlatlon
    
    def getPositionSphericalUnrotated(self) -> List[float]:
        r"""Get the position of the tile in spherical coordinates without taking into accounts earth rotation. For internal use only.
        
        Returns:
            List[float]: Position of the tile in spherical coordinates.
        """
        return self._positionsphericalunrotated

    def getPositionCartesian(self,scenario) -> List[float]:
        r"""Get the position of the tile in cartesian coordinates.
        
        Returns:
            List[float]: Position of the tile in cartesian coordinates."""
        pos = self.getPosition()    
        return EarthSurfacePosition.live(lon=pos[1], lat=pos[0], scenario=scenario)     

    def _update(self, update=True):
        self._updateSelf = update

    def _hasUpdate(self,) -> bool:
        return self._updateSelf


class FindUtils:
    @staticmethod
    def circleRegion(center: EarthSurfacePosition, radius_km: float, h3resolution: int) -> List[Tile]:
        r"""Return a list of tiles that are within a circular region around a given center point. Utitiliy function.
        
        Args:
            center (EarthSurfacePosition): Center of the circular region.
            radius_km (float): Radius of the circular region in kilometers.
            h3resolution (int): Resolution of the H3 tiles to be used.

        Returns:
            List[Tile]: List of tiles that are within the circular region.
        """
        radius_meters = radius_km * 1000
        
        center_h3 = h3.geo_to_h3(center.lat, center.lon, h3resolution)
        
        result_tiles = []
        h3_indices_to_check = set([center_h3])
        checked_h3_indices = set()
        
        while h3_indices_to_check:
            current_h3_index = h3_indices_to_check.pop()
            if current_h3_index in checked_h3_indices:
                continue
            
            checked_h3_indices.add(current_h3_index)
            current_h3_center = h3.h3_to_geo(current_h3_index)
            distance_meters = FindUtils._haversine_distance(center.lat, center.lon, current_h3_center[0], current_h3_center[1])
            
            if distance_meters <= radius_meters:
                # Create a Tile object for each valid H3 index
                tile = Tile(h3index=current_h3_index)
                result_tiles.append(tile)
                
                # Add neighboring H3 indices to the list of indices to check
                neighbors = h3.hex_ring(current_h3_index, 1)
                for neighbor in neighbors:
                    if neighbor not in checked_h3_indices:
                        h3_indices_to_check.add(neighbor)
        
        return result_tiles

    @staticmethod
    def landmass(h3resolution: int) -> List[Tile]:
        r"""Return a list of all tiles that are part of the earth's landmass at the given resolution.
        
        Args:
            h3resolution (int): Resolution of the H3 tiles to be used.

        Returns:
            List[Tile]: List of all landmass tiles that are part of the earth's landmass at the given resolution.
        """
        alltiles = FindUtils.getAllTilesAtResolution(h3resolution=h3resolution)
        from PIL import Image
        image = Image.open('webvisualizer/assets/textures/water_4k.png')
        image_np = np.array(image)
        IMG_WIDTH = image_np.shape[1]-1
        IMG_HEIGHT = image_np.shape[0]-1
        land_tiles = []
        for tile in alltiles:
            [lat,lon] = h3.h3_to_geo(tile.getH3Index())
            x = int(np.round((lon+180.0)/360.0*IMG_WIDTH))
            y = int(np.round((-lat+90)/180*IMG_HEIGHT))
            if image_np[y,x,0] < 255:
                land_tiles.append(tile)
        return land_tiles

    @staticmethod
    def getAllTilesAtResolution(h3resolution: int) -> List[Tile]:
        r"""Return a list of all tiles at the given resolution.
        
        Args:
            h3resolution (int): Resolution of the H3 tiles to be used.
            
        Returns:
            List[Tile]: List of all tiles at the given resolution
        """
        def get_children_at_res(h3_index, target_res):
            result = []
            next_res = h3.h3_get_resolution(h3_index) + 1

            # Iterate over all direct children of the current index
            for child in h3.h3_to_children(h3_index, next_res):
                if next_res >= target_res:
                    result.append(Tile(h3index=child))
                else:
                    result.extend(get_children_at_res(child, target_res))
            
            return result
        
        base_cells = h3.get_res0_indexes()
        all_tiles = []

        # Collect tiles for the target resolution
        for index in base_cells:
            all_tiles.extend(get_children_at_res(index, h3resolution))
        
        return all_tiles


    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Internal helper function to calculate the haversine distance between two points on the Earth's surface.
        # Radius of the Earth in meters
        R = 6371000
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance
class TilingAPI(RequireEncode):
    r"""Class providing geospatial capabilities for planet Earth.
    
    Note:
    The Tiling API instance is provided by the scenario object and can be accessed using the scenario.getTilingAPI() method.
    """

    def __init__(self, planet):
        self._owner = planet
        self._registered_tiles: Dict[str, Tile] = {}
        # settings
        self._settings = {
            "transparent": False,
            "dataVolumeTopOnly": False,
        }

    def registerTiles(self, tiles: List[Tile]):
        r"""Register a list of tiles as selected by the user with the Tiling API.
        
        Args:
            tiles (List[Tile]): List of tiles to be registered.

        Note:
        
        If tiles are already registered, an error is raised.
        """
        for tile in tiles:
            h3_index = tile.getH3Index()
            if h3_index in self._registered_tiles:
                raise ValueError(f"Tile with H3 index {h3_index} is already registered.")
            self._registered_tiles[h3_index] = tile

    def getTiles(self,) -> Dict[str, Tile]:
        r"""Get all registered tiles to the TilingAPI.
        
        Returns:
            Dict[str, Tile]: Dictionary of registered tiles with the H3 index as the key.
        """
        return self._registered_tiles
    
    def setSettings(self, transparent: Optional[bool]=None, dataVolumeTopOnly: Optional[bool]=None):
        r"""Set global settings related to rendering of the Tiling API.
        
        Args:
            transparent (Optional[bool], optional): If True, the tiles are rendered with transparency (more expensive).
            dataVolumeTopOnly (Optional[bool], optional): If True, only the top of the 3D volume is rendered. This enables to render a floating surface.
        """
        if transparent != None:
            self._settings["transparent"] = transparent
        if dataVolumeTopOnly != None:
            self._settings["dataVolumeTopOnly"] = dataVolumeTopOnly

    def encode(self, messagetype: Message.Type) -> Dict:
        r"""Encode the Tiling API's data for transfer to the frontend."""
        # Internally used. Not to be called by the user.
        if messagetype == Message.Type.SCENARIOSETUP:
            # transfer all tiles with all data
            return self._encodeTilesBinary(onlyupdates=False, settings=True)
        elif messagetype == Message.Type.PROPAGATIONDATA:
            # transfer only updated tiles with all data
            return self._encodeTilesBinary(onlyupdates=True)
        return {}
    
    def _encodeTilesBinary(self, onlyupdates=False, settings=False) -> Dict[str, List]:
        r"""Encode the tiles and their data in binary format for efficient transfer to the frontend.
        
        Note:
        Tile Data is efficiently encoded in binary format and transfered using base64 strings via websocket.
        """
        tiles_h3index_int64 = []    # Store tile index as int64 (native format for H3)
        tiles_colors_int32 = []     # Store tile color as int32 (4 bytes, 1 byte per color channel)
        # data
        tiles_data_heights_float32 = []     # Store tile height as float32 (4 bytes)
        tiles_data_colors_int32 = []        # Store tile data color as int32 (4 bytes, 1 byte per color channel)
        tiles_data_volumes_uint8 = []       # Store tile data volume as uint8 (1 byte)
        hasdata = False # Flag to indicate if any tile has additional data (height, volume, data color), if not, don't transfer.
        for tile in self._registered_tiles.values():
            if not onlyupdates or tile._hasUpdate():
                tiles_h3index_int64.append(tile._h3indexInt64)
                tiles_colors_int32.append(tile.getColor())
                tiles_data_heights_float32.append(tile.getData())
                tiles_data_colors_int32.append(tile.getDataColor())
                tiles_data_volumes_uint8.append(tile.getDataVolume())
                hasdata = hasdata or tile.getData() != np.float32(0)
                tile._update(update = False)

        def list_to_base64_int64(int64_list: List[int]) -> str:
            # Helper function to convert a list of int64 to a base64 string
            byte_array = b''.join(struct.pack('>q', x) for x in int64_list)  # '>q' is for big-endian int64
            return base64.b64encode(byte_array).decode('utf-8')
        
        def list_to_base64_int32(int32_list: List[int]) -> str:
            # Helper function to convert a list of int32 to a base64 string
            byte_array = b''.join(struct.pack('>i', x) for x in int32_list)  # '>i' is for big-endian int32
            return base64.b64encode(byte_array).decode('utf-8')
        
        def list_to_base64_float32(float32_list: List[np.float32]) -> str:            
            # Helper function to convert a list of float32 to a base64 string
            byte_array = b''.join(struct.pack('>f', x) for x in float32_list)  # '>f' is for big-endian float32
            return base64.b64encode(byte_array).decode('utf-8')

        def list_to_base64_uint8(uint8_list: List[np.uint8]) -> str:
            # Helper function to convert a list of uiont8 to a base64 string
            byte_array = b''.join(struct.pack('>B', x) for x in uint8_list)  # '>B' is for big-endian uint8
            return base64.b64encode(byte_array).decode('utf-8')

        # Pack data into a dict for encapsulation in JSON.
        data = {
            "tiles_h3index_int64": list_to_base64_int64(tiles_h3index_int64),
            "tiles_colors_int32": list_to_base64_int32(tiles_colors_int32),
            **({"hasData": True} if hasdata else {}),
            **({"tiles_data_heights_float32": list_to_base64_float32(tiles_data_heights_float32)} if hasdata else {}),
            **({"tiles_data_colors_int32": list_to_base64_int32(tiles_data_colors_int32)} if hasdata else {}),
            **({"tiles_data_volumes_uint8": list_to_base64_uint8(tiles_data_volumes_uint8)} if hasdata else {}),
            **({"settings": self._settings} if settings != None else {}),
        }
        # Only return dict if data is available.
        return data if len(tiles_h3index_int64) > 0 else None

    def find(self,) -> FindUtils:
        r"""Get a FindUtils instance. Convenience function.
        
        Returns:
            FindUtils: FindUtils object for the Tiling API.
        """
        return FindUtils()    
    
    
    class TileColor():
        r"""Class providing utility functions for color encoding of tiles."""
        @staticmethod
        def fromRGB(r: Optional[np.uint8] = 0, g: Optional[np.uint8] = 0, b: Optional[np.uint8] = 0, a: Optional[np.uint8] = 255) -> np.int32:
            r"""Create a 32-bit integer color value from RGBA values.
            
            Args:
                r (Optional[np.uint8], optional): Red channel value. Defaults to 0.
                g (Optional[np.uint8], optional): Green channel value. Defaults to 0.
                b (Optional[np.uint8], optional): Blue channel value. Defaults to 0.
                a (Optional[np.uint8], optional): Alpha channel value. Defaults to 255.

            Returns:
                np.int32: 32-bit integer color value.
            """
            # Ensure defaults are within valid range for uint8
            r = np.uint8(r) if r is not None else 0
            g = np.uint8(g) if g is not None else 0
            b = np.uint8(b) if b is not None else 0
            a = np.uint8(a) if a is not None else 255
            
            # Convert values to uint8 to ensure correct range and type
            r = np.uint8(r)
            g = np.uint8(g)
            b = np.uint8(b)
            a = np.uint8(a)
            
            # Combine RGBA values into a single int32 value
            color = (a << 24) | (r << 16) | (g << 8) | b
            return np.int32(color)
        
        @staticmethod
        def random(a: Optional[np.uint8] = None) -> np.int32:
            r"""Create a random 32-bit integer color value.
            
            Args:
                a (Optional[np.uint8], optional): Alpha channel value. Defaults to a random value if not supplied.

            Returns:
                np.int32: 32-bit integer color value.
            """
            r = np.uint8(np.random.randint(256))
            g = np.uint8(np.random.randint(256))
            b = np.uint8(np.random.randint(256))
            a = np.uint8(a if a != None else np.random.randint(256))
            
            # Combine RGBA values into a single int32 value
            color = (a << 24) | (r << 16) | (g << 8) | b
            return np.int32(color)