# Expo SDK and Services

## Table of Contents
- [Expo Setup and Configuration](#expo-setup-and-configuration)
- [Core Expo APIs](#core-expo-apis)
- [Device Features](#device-features)
- [Media and Files](#media-and-files)
- [Networking and Storage](#networking-and-storage)
- [Push Notifications](#push-notifications)
- [Authentication Services](#authentication-services)
- [Build and Deployment](#build-and-deployment)

## Expo Setup and Configuration

### Project Setup
```bash
# Install Expo CLI
npm install -g @expo/cli

# Create new Expo project
npx create-expo-app MyApp --template

# Templates available:
# blank - Minimal Expo template
# blank-typescript - Blank template with TypeScript
# tabs - Tab-based navigation template
# bare-minimum - Bare React Native project

cd MyApp

# Start development server
npx expo start

# Run on specific platform
npx expo start --ios
npx expo start --android
npx expo start --web
```

### App Configuration (app.json/app.config.js)
```javascript
// app.config.js
export default {
  expo: {
    name: "My Awesome App",
    slug: "my-awesome-app",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/icon.png",
    userInterfaceStyle: "light",
    splash: {
      image: "./assets/splash.png",
      resizeMode: "contain",
      backgroundColor: "#ffffff"
    },
    updates: {
      fallbackToCacheTimeout: 0,
      url: "https://u.expo.dev/your-project-id"
    },
    assetBundlePatterns: [
      "**/*"
    ],
    ios: {
      supportsTablet: true,
      bundleIdentifier: "com.yourcompany.myawesomeapp",
      buildNumber: "1.0.0",
      infoPlist: {
        NSCameraUsageDescription: "This app uses the camera to take photos.",
        NSLocationWhenInUseUsageDescription: "This app uses location to show nearby places."
      }
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./assets/adaptive-icon.png",
        backgroundColor: "#FFFFFF"
      },
      package: "com.yourcompany.myawesomeapp",
      versionCode: 1,
      permissions: [
        "CAMERA",
        "ACCESS_FINE_LOCATION",
        "WRITE_EXTERNAL_STORAGE"
      ]
    },
    web: {
      favicon: "./assets/favicon.png"
    },
    plugins: [
      "expo-camera",
      "expo-location",
      [
        "expo-notifications",
        {
          icon: "./assets/notification-icon.png",
          color: "#ffffff"
        }
      ]
    ],
    extra: {
      apiUrl: process.env.API_URL || "https://api.example.com",
      eas: {
        projectId: "your-project-id"
      }
    }
  }
};
```

## Core Expo APIs

### Constants and System Info
```javascript
import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Constants from 'expo-constants';
import * as Device from 'expo-device';
import * as Application from 'expo-application';

function SystemInfoScreen() {
  const [systemInfo, setSystemInfo] = useState({});

  useEffect(() => {
    getSystemInfo();
  }, []);

  const getSystemInfo = async () => {
    const info = {
      // Constants
      expoVersion: Constants.expoVersion,
      statusBarHeight: Constants.statusBarHeight,
      platform: Constants.platform,
      
      // Device info
      deviceName: Device.deviceName,
      deviceType: Device.deviceType,
      isDevice: Device.isDevice,
      brand: Device.brand,
      manufacturer: Device.manufacturer,
      modelName: Device.modelName,
      osName: Device.osName,
      osVersion: Device.osVersion,
      
      // Application info
      applicationName: Application.applicationName,
      applicationVersion: Application.nativeApplicationVersion,
      buildVersion: Application.nativeBuildVersion,
      applicationId: Application.applicationId,
    };

    setSystemInfo(info);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>System Information</Text>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Expo</Text>
        <Text>Version: {systemInfo.expoVersion}</Text>
        <Text>Status Bar Height: {systemInfo.statusBarHeight}</Text>
        <Text>Platform: {JSON.stringify(systemInfo.platform)}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Device</Text>
        <Text>Name: {systemInfo.deviceName}</Text>
        <Text>Type: {systemInfo.deviceType}</Text>
        <Text>Is Physical Device: {systemInfo.isDevice ? 'Yes' : 'No'}</Text>
        <Text>Brand: {systemInfo.brand}</Text>
        <Text>Manufacturer: {systemInfo.manufacturer}</Text>
        <Text>Model: {systemInfo.modelName}</Text>
        <Text>OS: {systemInfo.osName} {systemInfo.osVersion}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Application</Text>
        <Text>Name: {systemInfo.applicationName}</Text>
        <Text>Version: {systemInfo.applicationVersion}</Text>
        <Text>Build: {systemInfo.buildVersion}</Text>
        <Text>Bundle ID: {systemInfo.applicationId}</Text>
      </View>
    </View>
  );
}
```

### Screen Orientation and Dimensions
```javascript
import React, { useEffect, useState } from 'react';
import { View, Text, Button, Dimensions } from 'react-native';
import * as ScreenOrientation from 'expo-screen-orientation';

function OrientationExample() {
  const [orientation, setOrientation] = useState(null);
  const [dimensions, setDimensions] = useState(Dimensions.get('window'));

  useEffect(() => {
    // Get initial orientation
    ScreenOrientation.getOrientationAsync().then(setOrientation);

    // Listen for orientation changes
    const subscription = ScreenOrientation.addOrientationChangeListener(
      (event) => {
        setOrientation(event.orientationInfo.orientation);
      }
    );

    // Listen for dimension changes
    const dimensionSubscription = Dimensions.addEventListener(
      'change',
      ({ window }) => {
        setDimensions(window);
      }
    );

    return () => {
      subscription.remove();
      dimensionSubscription?.remove();
    };
  }, []);

  const lockOrientation = async (orientationLock) => {
    await ScreenOrientation.lockAsync(orientationLock);
  };

  const unlockOrientation = async () => {
    await ScreenOrientation.unlockAsync();
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Screen Orientation</Text>
      
      <Text>Current Orientation: {orientation}</Text>
      <Text>Width: {dimensions.width}</Text>
      <Text>Height: {dimensions.height}</Text>
      <Text>Scale: {dimensions.scale}</Text>

      <View style={styles.buttonContainer}>
        <Button
          title="Lock Portrait"
          onPress={() => lockOrientation(ScreenOrientation.OrientationLock.PORTRAIT)}
        />
        <Button
          title="Lock Landscape"
          onPress={() => lockOrientation(ScreenOrientation.OrientationLock.LANDSCAPE)}
        />
        <Button
          title="Unlock"
          onPress={unlockOrientation}
        />
      </View>
    </View>
  );
}
```

## Device Features

### Camera Integration
```javascript
import React, { useState, useRef } from 'react';
import { View, Text, TouchableOpacity, Image, StyleSheet } from 'react-native';
import { Camera } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';

function CameraScreen() {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [flash, setFlash] = useState(Camera.Constants.FlashMode.off);
  const [photo, setPhoto] = useState(null);
  const [isPreview, setIsPreview] = useState(false);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const takePicture = async () => {
    if (cameraRef.current) {
      const options = {
        quality: 0.8,
        base64: true,
        exif: false,
      };
      
      const photo = await cameraRef.current.takePictureAsync(options);
      setPhoto(photo);
      setIsPreview(true);
    }
  };

  const savePhoto = async () => {
    if (photo) {
      try {
        const { status } = await MediaLibrary.requestPermissionsAsync();
        if (status === 'granted') {
          await MediaLibrary.saveToLibraryAsync(photo.uri);
          alert('Photo saved to gallery!');
        }
      } catch (error) {
        console.error('Error saving photo:', error);
      }
    }
  };

  const retakePhoto = () => {
    setPhoto(null);
    setIsPreview(false);
  };

  if (hasPermission === null) {
    return <View />;
  }
  
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  if (isPreview && photo) {
    return (
      <View style={styles.container}>
        <Image source={{ uri: photo.uri }} style={styles.preview} />
        <View style={styles.previewButtons}>
          <TouchableOpacity style={styles.button} onPress={retakePhoto}>
            <Text style={styles.buttonText}>Retake</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={savePhoto}>
            <Text style={styles.buttonText}>Save</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera 
        style={styles.camera} 
        type={type}
        flashMode={flash}
        ref={cameraRef}
      >
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.button}
            onPress={() => {
              setType(
                type === Camera.Constants.Type.back
                  ? Camera.Constants.Type.front
                  : Camera.Constants.Type.back
              );
            }}
          >
            <Text style={styles.buttonText}>Flip Camera</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.button}
            onPress={() => {
              setFlash(
                flash === Camera.Constants.FlashMode.off
                  ? Camera.Constants.FlashMode.on
                  : Camera.Constants.FlashMode.off
              );
            }}
          >
            <Text style={styles.buttonText}>
              Flash: {flash === Camera.Constants.FlashMode.off ? 'Off' : 'On'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.captureButton} onPress={takePicture}>
            <View style={styles.captureButtonInner} />
          </TouchableOpacity>
        </View>
      </Camera>
    </View>
  );
}
```

### Location Services
```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import * as Location from 'expo-location';
import MapView, { Marker } from 'react-native-maps';

function LocationScreen() {
  const [location, setLocation] = useState(null);
  const [address, setAddress] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isWatching, setIsWatching] = useState(false);
  const [watchSubscription, setWatchSubscription] = useState(null);

  useEffect(() => {
    return () => {
      if (watchSubscription) {
        watchSubscription.remove();
      }
    };
  }, [watchSubscription]);

  const getCurrentLocation = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        return;
      }

      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });
      
      setLocation(location);
      
      // Reverse geocoding
      const address = await Location.reverseGeocodeAsync({
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
      });
      
      setAddress(address[0]);
    } catch (error) {
      setErrorMsg('Error getting location: ' + error.message);
    }
  };

  const watchLocation = async () => {
    if (isWatching) {
      if (watchSubscription) {
        watchSubscription.remove();
        setWatchSubscription(null);
      }
      setIsWatching(false);
      return;
    }

    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        return;
      }

      const subscription = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.High,
          timeInterval: 1000, // Update every second
          distanceInterval: 1, // Update every meter
        },
        (location) => {
          setLocation(location);
        }
      );

      setWatchSubscription(subscription);
      setIsWatching(true);
    } catch (error) {
      setErrorMsg('Error watching location: ' + error.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Location Services</Text>

      {errorMsg && <Text style={styles.error}>{errorMsg}</Text>}

      <View style={styles.buttonContainer}>
        <Button title="Get Current Location" onPress={getCurrentLocation} />
        <Button 
          title={isWatching ? "Stop Watching" : "Watch Location"} 
          onPress={watchLocation}
        />
      </View>

      {location && (
        <View style={styles.locationInfo}>
          <Text style={styles.subtitle}>Current Location:</Text>
          <Text>Latitude: {location.coords.latitude.toFixed(6)}</Text>
          <Text>Longitude: {location.coords.longitude.toFixed(6)}</Text>
          <Text>Accuracy: {location.coords.accuracy}m</Text>
          <Text>Altitude: {location.coords.altitude}m</Text>
          <Text>Speed: {location.coords.speed}m/s</Text>
          <Text>Heading: {location.coords.heading}°</Text>
          <Text>Timestamp: {new Date(location.timestamp).toLocaleString()}</Text>
        </View>
      )}

      {address && (
        <View style={styles.addressInfo}>
          <Text style={styles.subtitle}>Address:</Text>
          <Text>{address.street} {address.streetNumber}</Text>
          <Text>{address.city}, {address.region}</Text>
          <Text>{address.postalCode}, {address.country}</Text>
        </View>
      )}

      {location && (
        <MapView
          style={styles.map}
          region={{
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
            latitudeDelta: 0.01,
            longitudeDelta: 0.01,
          }}
        >
          <Marker
            coordinate={{
              latitude: location.coords.latitude,
              longitude: location.coords.longitude,
            }}
            title="Current Location"
            description={address ? `${address.street}, ${address.city}` : 'You are here'}
          />
        </MapView>
      )}
    </View>
  );
}
```

### Sensors
```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Accelerometer, Gyroscope, Magnetometer, Barometer } from 'expo-sensors';

function SensorsScreen() {
  const [accelerometerData, setAccelerometerData] = useState({});
  const [gyroscopeData, setGyroscopeData] = useState({});
  const [magnetometerData, setMagnetometerData] = useState({});
  const [barometerData, setBarometerData] = useState({});

  useEffect(() => {
    // Set update intervals
    Accelerometer.setUpdateInterval(100);
    Gyroscope.setUpdateInterval(100);
    Magnetometer.setUpdateInterval(100);
    Barometer.setUpdateInterval(1000);

    // Subscribe to sensors
    const accelerometerSubscription = Accelerometer.addListener(setAccelerometerData);
    const gyroscopeSubscription = Gyroscope.addListener(setGyroscopeData);
    const magnetometerSubscription = Magnetometer.addListener(setMagnetometerData);
    const barometerSubscription = Barometer.addListener(setBarometerData);

    return () => {
      accelerometerSubscription && accelerometerSubscription.remove();
      gyroscopeSubscription && gyroscopeSubscription.remove();
      magnetometerSubscription && magnetometerSubscription.remove();
      barometerSubscription && barometerSubscription.remove();
    };
  }, []);

  const round = (n) => {
    if (!n) return 0;
    return Math.floor(n * 100) / 100;
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Device Sensors</Text>

      <View style={styles.sensorContainer}>
        <Text style={styles.sensorTitle}>Accelerometer</Text>
        <Text>X: {round(accelerometerData.x)}</Text>
        <Text>Y: {round(accelerometerData.y)}</Text>
        <Text>Z: {round(accelerometerData.z)}</Text>
      </View>

      <View style={styles.sensorContainer}>
        <Text style={styles.sensorTitle}>Gyroscope</Text>
        <Text>X: {round(gyroscopeData.x)}</Text>
        <Text>Y: {round(gyroscopeData.y)}</Text>
        <Text>Z: {round(gyroscopeData.z)}</Text>
      </View>

      <View style={styles.sensorContainer}>
        <Text style={styles.sensorTitle}>Magnetometer</Text>
        <Text>X: {round(magnetometerData.x)}</Text>
        <Text>Y: {round(magnetometerData.y)}</Text>
        <Text>Z: {round(magnetometerData.z)}</Text>
      </View>

      <View style={styles.sensorContainer}>
        <Text style={styles.sensorTitle}>Barometer</Text>
        <Text>Pressure: {round(barometerData.pressure)} hPa</Text>
        <Text>Relative Altitude: {round(barometerData.relativeAltitude)} m</Text>
      </View>
    </View>
  );
}
```

## Media and Files

### Image Picker
```javascript
import React, { useState } from 'react';
import { View, Text, Image, Button, StyleSheet, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';

function ImagePickerScreen() {
  const [image, setImage] = useState(null);
  const [manipulatedImage, setManipulatedImage] = useState(null);

  const pickImage = async (useCamera = false) => {
    const permissionResult = useCamera 
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      Alert.alert('Permission denied', 'Permission to access camera/gallery is required!');
      return;
    }

    const options = {
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
      base64: true,
      exif: true,
    };

    const result = useCamera 
      ? await ImagePicker.launchCameraAsync(options)
      : await ImagePicker.launchImageLibraryAsync(options);

    if (!result.canceled) {
      setImage(result.assets[0]);
      setManipulatedImage(null);
    }
  };

  const manipulateImage = async () => {
    if (!image) return;

    try {
      const manipResult = await ImageManipulator.manipulateAsync(
        image.uri,
        [
          { resize: { width: 300 } },
          { rotate: 90 },
          { crop: { originX: 0, originY: 0, width: 300, height: 300 } },
        ],
        { 
          compress: 0.7, 
          format: ImageManipulator.SaveFormat.JPEG,
          base64: true,
        }
      );
      
      setManipulatedImage(manipResult);
    } catch (error) {
      Alert.alert('Error', 'Failed to manipulate image: ' + error.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Image Picker & Manipulation</Text>

      <View style={styles.buttonContainer}>
        <Button title="Pick from Gallery" onPress={() => pickImage(false)} />
        <Button title="Take Photo" onPress={() => pickImage(true)} />
      </View>

      {image && (
        <View style={styles.imageContainer}>
          <Text style={styles.subtitle}>Original Image:</Text>
          <Image source={{ uri: image.uri }} style={styles.image} />
          <Text>Size: {image.width}x{image.height}</Text>
          <Text>File Size: {(image.fileSize / 1024).toFixed(2)} KB</Text>
          
          <Button title="Manipulate Image" onPress={manipulateImage} />
        </View>
      )}

      {manipulatedImage && (
        <View style={styles.imageContainer}>
          <Text style={styles.subtitle}>Manipulated Image:</Text>
          <Image source={{ uri: manipulatedImage.uri }} style={styles.image} />
          <Text>Size: {manipulatedImage.width}x{manipulatedImage.height}</Text>
        </View>
      )}
    </View>
  );
}
```

### File System
```javascript
import React, { useState } from 'react';
import { View, Text, Button, StyleSheet, Alert } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

function FileSystemScreen() {
  const [fileInfo, setFileInfo] = useState(null);
  const [directoryContents, setDirectoryContents] = useState([]);

  const createFile = async () => {
    try {
      const fileUri = FileSystem.documentDirectory + 'sample.txt';
      const content = 'Hello, this is a sample file created by Expo FileSystem!';
      
      await FileSystem.writeAsStringAsync(fileUri, content);
      
      const info = await FileSystem.getInfoAsync(fileUri);
      setFileInfo(info);
      
      Alert.alert('Success', 'File created successfully!');
    } catch (error) {
      Alert.alert('Error', 'Failed to create file: ' + error.message);
    }
  };

  const readFile = async () => {
    try {
      const fileUri = FileSystem.documentDirectory + 'sample.txt';
      const content = await FileSystem.readAsStringAsync(fileUri);
      
      Alert.alert('File Content', content);
    } catch (error) {
      Alert.alert('Error', 'Failed to read file: ' + error.message);
    }
  };

  const downloadFile = async () => {
    try {
      const downloadUri = FileSystem.documentDirectory + 'downloaded_image.jpg';
      const download = await FileSystem.downloadAsync(
        'https://picsum.photos/200/300',
        downloadUri
      );
      
      const info = await FileSystem.getInfoAsync(download.uri);
      setFileInfo(info);
      
      Alert.alert('Success', 'File downloaded successfully!');
    } catch (error) {
      Alert.alert('Error', 'Failed to download file: ' + error.message);
    }
  };

  const shareFile = async () => {
    try {
      const fileUri = FileSystem.documentDirectory + 'sample.txt';
      const isAvailable = await Sharing.isAvailableAsync();
      
      if (isAvailable) {
        await Sharing.shareAsync(fileUri);
      } else {
        Alert.alert('Error', 'Sharing is not available on this device');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to share file: ' + error.message);
    }
  };

  const listDirectory = async () => {
    try {
      const contents = await FileSystem.readDirectoryAsync(FileSystem.documentDirectory);
      setDirectoryContents(contents);
    } catch (error) {
      Alert.alert('Error', 'Failed to list directory: ' + error.message);
    }
  };

  const deleteFile = async () => {
    try {
      const fileUri = FileSystem.documentDirectory + 'sample.txt';
      await FileSystem.deleteAsync(fileUri);
      
      setFileInfo(null);
      Alert.alert('Success', 'File deleted successfully!');
    } catch (error) {
      Alert.alert('Error', 'Failed to delete file: ' + error.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>File System Operations</Text>

      <View style={styles.buttonContainer}>
        <Button title="Create File" onPress={createFile} />
        <Button title="Read File" onPress={readFile} />
        <Button title="Download File" onPress={downloadFile} />
        <Button title="Share File" onPress={shareFile} />
        <Button title="List Directory" onPress={listDirectory} />
        <Button title="Delete File" onPress={deleteFile} />
      </View>

      {fileInfo && (
        <View style={styles.infoContainer}>
          <Text style={styles.subtitle}>File Info:</Text>
          <Text>URI: {fileInfo.uri}</Text>
          <Text>Size: {fileInfo.size} bytes</Text>
          <Text>Exists: {fileInfo.exists ? 'Yes' : 'No'}</Text>
          <Text>Is Directory: {fileInfo.isDirectory ? 'Yes' : 'No'}</Text>
          <Text>Modified: {new Date(fileInfo.modificationTime * 1000).toLocaleString()}</Text>
        </View>
      )}

      {directoryContents.length > 0 && (
        <View style={styles.infoContainer}>
          <Text style={styles.subtitle}>Directory Contents:</Text>
          {directoryContents.map((item, index) => (
            <Text key={index}>• {item}</Text>
          ))}
        </View>
      )}

      <View style={styles.infoContainer}>
        <Text style={styles.subtitle}>Directory Paths:</Text>
        <Text>Document: {FileSystem.documentDirectory}</Text>
        <Text>Cache: {FileSystem.cacheDirectory}</Text>
        <Text>Bundle: {FileSystem.bundleDirectory}</Text>
      </View>
    </View>
  );
}
```

## Networking and Storage

### Secure Store
```javascript
import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Alert } from 'react-native';
import * as SecureStore from 'expo-secure-store';

function SecureStoreScreen() {
  const [key, setKey] = useState('');
  const [value, setValue] = useState('');
  const [retrievedValue, setRetrievedValue] = useState('');

  const saveSecurely = async () => {
    if (!key || !value) {
      Alert.alert('Error', 'Please enter both key and value');
      return;
    }

    try {
      await SecureStore.setItemAsync(key, value);
      Alert.alert('Success', 'Data saved securely!');
      setKey('');
      setValue('');
    } catch (error) {
      Alert.alert('Error', 'Failed to save data: ' + error.message);
    }
  };

  const retrieveSecurely = async () => {
    if (!key) {
      Alert.alert('Error', 'Please enter a key');
      return;
    }

    try {
      const result = await SecureStore.getItemAsync(key);
      if (result) {
        setRetrievedValue(result);
        Alert.alert('Success', 'Data retrieved successfully!');
      } else {
        Alert.alert('Not Found', 'No data found for this key');
        setRetrievedValue('');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to retrieve data: ' + error.message);
    }
  };

  const deleteSecurely = async () => {
    if (!key) {
      Alert.alert('Error', 'Please enter a key');
      return;
    }

    try {
      await SecureStore.deleteItemAsync(key);
      Alert.alert('Success', 'Data deleted successfully!');
      setRetrievedValue('');
    } catch (error) {
      Alert.alert('Error', 'Failed to delete data: ' + error.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Secure Store</Text>

      <TextInput
        style={styles.input}
        placeholder="Enter key"
        value={key}
        onChangeText={setKey}
      />

      <TextInput
        style={styles.input}
        placeholder="Enter value"
        value={value}
        onChangeText={setValue}
        multiline
      />

      <View style={styles.buttonContainer}>
        <Button title="Save" onPress={saveSecurely} />
        <Button title="Retrieve" onPress={retrieveSecurely} />
        <Button title="Delete" onPress={deleteSecurely} />
      </View>

      {retrievedValue !== '' && (
        <View style={styles.resultContainer}>
          <Text style={styles.subtitle}>Retrieved Value:</Text>
          <Text>{retrievedValue}</Text>
        </View>
      )}
    </View>
  );
}
```

## Push Notifications

### Notification Setup
```javascript
import React, { useState, useEffect, useRef } from 'react';
import { View, Text, Button, Platform, StyleSheet } from 'react-native';
import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';

// Configure notification handler
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

function NotificationScreen() {
  const [expoPushToken, setExpoPushToken] = useState('');
  const [notification, setNotification] = useState(false);
  const notificationListener = useRef();
  const responseListener = useRef();

  useEffect(() => {
    registerForPushNotificationsAsync().then(token => setExpoPushToken(token));

    // Listen for notifications
    notificationListener.current = Notifications.addNotificationReceivedListener(notification => {
      setNotification(notification);
    });

    // Listen for notification responses
    responseListener.current = Notifications.addNotificationResponseReceivedListener(response => {
      console.log('Notification response:', response);
    });

    return () => {
      Notifications.removeNotificationSubscription(notificationListener.current);
      Notifications.removeNotificationSubscription(responseListener.current);
    };
  }, []);

  const registerForPushNotificationsAsync = async () => {
    let token;
    
    if (Device.isDevice) {
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;
      
      if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
      }
      
      if (finalStatus !== 'granted') {
        alert('Failed to get push token for push notification!');
        return;
      }
      
      token = (await Notifications.getExpoPushTokenAsync()).data;
    } else {
      alert('Must use physical device for Push Notifications');
    }

    if (Platform.OS === 'android') {
      Notifications.setNotificationChannelAsync('default', {
        name: 'default',
        importance: Notifications.AndroidImportance.MAX,
        vibrationPattern: [0, 250, 250, 250],
        lightColor: '#FF231F7C',
      });
    }

    return token;
  };

  const sendLocalNotification = async () => {
    await Notifications.scheduleNotificationAsync({
      content: {
        title: "Local Notification",
        body: 'This is a local notification!',
        data: { data: 'goes here' },
      },
      trigger: { seconds: 2 },
    });
  };

  const sendScheduledNotification = async () => {
    await Notifications.scheduleNotificationAsync({
      content: {
        title: "Scheduled Notification",
        body: 'This notification was scheduled!',
        data: { type: 'scheduled' },
      },
      trigger: {
        seconds: 60,
        repeats: false,
      },
    });
  };

  const sendDailyNotification = async () => {
    await Notifications.scheduleNotificationAsync({
      content: {
        title: "Daily Reminder",
        body: 'This is your daily reminder!',
        data: { type: 'daily' },
      },
      trigger: {
        hour: 9,
        minute: 0,
        repeats: true,
      },
    });
  };

  const cancelAllNotifications = async () => {
    await Notifications.cancelAllScheduledNotificationsAsync();
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Push Notifications</Text>

      <Text style={styles.token}>
        Push Token: {expoPushToken}
      </Text>

      <View style={styles.buttonContainer}>
        <Button
          title="Send Local Notification"
          onPress={sendLocalNotification}
        />
        <Button
          title="Schedule Notification (1 min)"
          onPress={sendScheduledNotification}
        />
        <Button
          title="Schedule Daily Notification"
          onPress={sendDailyNotification}
        />
        <Button
          title="Cancel All Notifications"
          onPress={cancelAllNotifications}
        />
      </View>

      {notification && (
        <View style={styles.notificationContainer}>
          <Text style={styles.subtitle}>Last Notification:</Text>
          <Text>Title: {notification.request.content.title}</Text>
          <Text>Body: {notification.request.content.body}</Text>
          <Text>Data: {JSON.stringify(notification.request.content.data)}</Text>
        </View>
      )}
    </View>
  );
}

// Send push notification to specific device
async function sendPushNotification(expoPushToken, title, body, data = {}) {
  const message = {
    to: expoPushToken,
    sound: 'default',
    title: title,
    body: body,
    data: data,
  };

  await fetch('https://exp.host/--/api/v2/push/send', {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Accept-encoding': 'gzip, deflate',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(message),
  });
}
```

## Build and Deployment

### EAS Build Configuration
```javascript
// eas.json
{
  "cli": {
    "version": ">= 3.0.0"
  },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal",
      "ios": {
        "resourceClass": "m1-medium"
      }
    },
    "preview": {
      "distribution": "internal",
      "ios": {
        "simulator": true
      }
    },
    "production": {
      "ios": {
        "resourceClass": "m1-medium"
      }
    }
  },
  "submit": {
    "production": {
      "ios": {
        "appleId": "your-apple-id@example.com",
        "ascAppId": "1234567890",
        "appleTeamId": "ABCD123456"
      },
      "android": {
        "serviceAccountKeyPath": "../path/to/api-key.json",
        "track": "internal"
      }
    }
  }
}
```

### Build Commands
```bash
# Install EAS CLI
npm install -g eas-cli

# Configure project
eas build:configure

# Build for development
eas build --platform ios --profile development
eas build --platform android --profile development

# Build for production
eas build --platform ios --profile production
eas build --platform android --profile production

# Build for both platforms
eas build --platform all --profile production

# Submit to stores
eas submit --platform ios
eas submit --platform android

# Update app (OTA updates)
eas update --branch production --message "Bug fixes and improvements"
```

---

*Continue to: [14-react-native-apis.md](./14-react-native-apis.md)*
