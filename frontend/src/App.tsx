import React, { useState, useRef, useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, useMap, useMapEvent } from 'react-leaflet'; // Add useMapEvent
import { Upload, Search, Loader2 } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import { Icon, LatLngBounds, LatLng } from 'leaflet';
import currentLocationIconUrl from './current-location-icon.png'; // Add this line
import criminalIconUrl from './criminal-icon.png'; // Add this line

// Fix for default marker icon
delete (Icon.Default.prototype as any)._getIconUrl;
Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface Detection {
  metadata: {
    cctv: number;
    loc: {
      latitude: number;
      longitude: number;
    };
    time: string;
  };
}

interface DetectionResponse {
  detection_id: string;
  results: Detection[];
}

interface PlateDetection {
  metadata: {
    cctv: number;
    loc: {
      latitude: number;
      longitude: number;
    };
    time: string;
  };
  car_number: string;
  match_count: number;
}

interface PlateDetectionResponse {
  detection_id: string;
  plate_detections: PlateDetection[];
}

// Component to update map bounds
function MapBounds({ coordinates }: { coordinates: { lat: number; lng: number }[] }) {
  const map = useMap();

  useEffect(() => {
    if (coordinates.length > 0) {
      const bounds = new LatLngBounds(
        coordinates.map(coord => new LatLng(coord.lat, coord.lng))
      );
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [coordinates, map]);

  return null;
}

// Component to fetch route-based path
function RoutePath({ coordinates }: { coordinates: { lat: number; lng: number }[] }) {
  const map = useMap();

  useEffect(() => {
    if (coordinates.length > 1) {
      const fetchRoute = async () => {
        const waypoints = coordinates.map(coord => `${coord.lng},${coord.lat}`).join(';');
        const response = await fetch(`https://router.project-osrm.org/route/v1/driving/${waypoints}?overview=full&geometries=geojson`);
        const data = await response.json();
        const route = data.routes[0].geometry.coordinates.map((coord: [number, number]) => ({
          lat: coord[1],
          lng: coord[0],
        }));
        map.fitBounds(route.map(coord => new LatLng(coord.lat, coord.lng)), { padding: [50, 50] });
        setRoutePath(route);
      };

      fetchRoute();
    }
  }, [coordinates, map]);

  const [routePath, setRoutePath] = useState<{ lat: number; lng: number }[]>([]);

  return routePath.length > 0 ? (
    <Polyline
      positions={routePath}
      color="blue"
      weight={3}
      opacity={0.7}
    />
  ) : null;
}

function App() {
  const [isUploading, setIsUploading] = useState(false);
  const [isTracking, setIsTracking] = useState(false);
  const [detectionId, setDetectionId] = useState<string | null>(null);
  const [carNumber, setCarNumber] = useState('');
  const [carColor, setCarColor] = useState('');
  const [detections, setDetections] = useState<PlateDetection[]>([]);
  const [currentLocation, setCurrentLocation] = useState<{ lat: number; lng: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [mapKey, setMapKey] = useState(0); // Add key for map remounting

  const carColors = [
    'white', 'black', 'red', 'blue', 'silver', 'gray', 
    'green', 'yellow', 'brown', 'gold', 'orange'
  ];

  useEffect(() => {
    let mounted = true;

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          if (mounted) {
            setCurrentLocation({
              lat: position.coords.latitude,
              lng: position.coords.longitude,
            });
            setMapKey(prev => prev + 1); // Remount map when location is set
          }
        },
        (error) => {
          console.error('Error getting location:', error);
          if (mounted) {
            setCurrentLocation({ lat: 23.8103, lng: 90.4125 }); // Default to Dhaka
            setMapKey(prev => prev + 1);
          }
        }
      );
    }

    return () => {
      mounted = false;
    };
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('car_color', carColor);
    formData.append('car_number', carNumber);

    try {
      const response = await fetch('http://103.121.217.35:8080/detect_zip', {
        method: 'POST',
        body: formData,
      });

      const data: DetectionResponse = await response.json();
      setDetectionId(data.detection_id);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleTrack = async () => {
    if (!detectionId || !carNumber) return;

    setIsTracking(true);
    try {
      const response = await fetch('http://103.121.217.35:8080/detect_license_plate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          detection_id: detectionId,
          car_number: carNumber,
        }),
      });

      const data: PlateDetectionResponse = await response.json();
      const uniqueDetections = data.plate_detections.reduce((acc, detection) => {
        if (!acc.some(d => d.metadata.cctv === detection.metadata.cctv)) {
          acc.push(detection);
        }
        return acc;
      }, [] as PlateDetection[]);

      const topDetections = uniqueDetections
        .sort((a, b) => b.match_count - a.match_count)
        .slice(0, 5);

      setDetections(topDetections);
    } catch (error) {
      console.error('Error tracking car:', error);
      alert('Error tracking car. Please try again.');
    } finally {
      setIsTracking(false);
    }
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = '/defaultZip.zip'; // Path to the default ZIP file in the public folder
    link.download = 'defaultZip.zip';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const allCoordinates = React.useMemo(() => {
    const coords = detections.map(detection => ({
      lat: detection.metadata.loc.latitude,
      lng: detection.metadata.loc.longitude,
    }));

    if (currentLocation) {
      coords.unshift(currentLocation);
    }

    return coords;
  }, [detections, currentLocation]);

  const currentLocationIcon = new Icon({
    iconUrl: currentLocationIconUrl,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
  });

  const criminalIcon = new Icon({
    iconUrl: criminalIconUrl,
    iconSize: [50, 41],
    iconAnchor: [24, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
  });

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-2xl font-bold text-gray-800 mb-6">Car Detection System</h1>

          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="carColor" className="block text-sm font-medium text-gray-700 mb-2">
                  Car Color
                </label>
                <select
                  id="carColor"
                  value={carColor}
                  onChange={(e) => setCarColor(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select a color</option>
                  {carColors.map((color) => (
                    <option key={color} value={color}>
                      {color.charAt(0).toUpperCase() + color.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="carNumber" className="block text-sm font-medium text-gray-700 mb-2">
                  Car Number
                </label>
                <input
                  id="carNumber"
                  type="text"
                  value={carNumber}
                  onChange={(e) => setCarNumber(e.target.value)}
                  placeholder="Enter car number"
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Detection Files (ZIP)
              </label>
              <div className="flex items-center gap-4">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
                  disabled={isUploading || !carColor || !carNumber}
                >
                  {isUploading ? (
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  ) : (
                    <Upload className="w-5 h-5 mr-2" />
                  )}
                  {isUploading ? 'Uploading...' : 'Upload ZIP'}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".zip"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                {detectionId && (
                  <span className="text-sm text-green-600">Upload complete! Detection ID: {detectionId}</span>
                )}
                <button
                  onClick={handleDownload}
                  className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
                >
                  Download Default ZIP
                </button>
              </div>
            </div>

            {detectionId && (
              <div className="space-y-4">
                <div>
                  <div className="flex items-center gap-4">
                    <button
                      onClick={handleTrack}
                      disabled={isTracking || !carNumber}
                      className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50"
                    >
                      {isTracking ? (
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      ) : (
                        <Search className="w-5 h-5 mr-2" />
                      )}
                      {isTracking ? 'Tracking...' : 'Track Car'}
                    </button>
                  </div>
                </div>

                {detections.length > 0 && (
                  <div>
                    <h2 className="text-lg font-semibold mb-2">Detection Results:</h2>
                    <div className="space-y-2">
                      {detections.map((detection, index) => (
                        <div key={index} className="bg-gray-50 p-3 rounded-md">
                          <p className="text-sm text-gray-700">
                            CCTV: {detection.metadata.cctv} | 
                            Time: {detection.metadata.time} | 
                            Match Count: {detection.match_count}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-2" style={{ height: '600px' }}>
          {currentLocation && (
            <MapContainer
              key={mapKey}
              center={[currentLocation.lat, currentLocation.lng]}
              zoom={13}
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              <MapBounds coordinates={allCoordinates} />
              
              {/* Current location marker */}
              {currentLocation && (
                <Marker 
                  position={[currentLocation.lat, currentLocation.lng]}
                  icon={currentLocationIcon} // Use the custom icon
                >
                </Marker>
              )}

              {/* Detection markers */}
              {detections.map((detection, index) => (
                <Marker
                  key={index}
                  position={[detection.metadata.loc.latitude, detection.metadata.loc.longitude]}
                  icon={criminalIcon} // Use the criminal icon
                >
                </Marker>
              ))}

              {/* Route-based path */}
              <RoutePath coordinates={allCoordinates} />
            </MapContainer>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;