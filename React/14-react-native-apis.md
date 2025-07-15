# React Native APIs and Platform Features

## Table of Contents
- [Platform-Specific Code](#platform-specific-code)
- [Native Modules and Bridges](#native-modules-and-bridges)
- [Device APIs](#device-apis)
- [Networking](#networking)
- [Storage Solutions](#storage-solutions)
- [Background Tasks](#background-tasks)
- [Linking and Deep Links](#linking-and-deep-links)
- [Permissions](#permissions)

## Platform-Specific Code

### Platform Detection and Conditional Code
```javascript
import React from 'react';
import { Platform, StyleSheet, Text, View } from 'react-native';

function PlatformSpecificExample() {
  const platformName = Platform.OS;
  const platformVersion = Platform.Version;

  // Platform-specific component rendering
  const renderPlatformSpecific = () => {
    if (Platform.OS === 'ios') {
      return <Text>Running on iOS {Platform.Version}</Text>;
    } else if (Platform.OS === 'android') {
      return <Text>Running on Android API {Platform.Version}</Text>;
    } else {
      return <Text>Running on {Platform.OS}</Text>;
    }
  };

  // Platform.select for conditional values
  const platformStyles = Platform.select({
    ios: {
      backgroundColor: '#007AFF',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.25,
      shadowRadius: 3.84,
    },
    android: {
      backgroundColor: '#4CAF50',
      elevation: 8,
    },
    default: {
      backgroundColor: '#888',
    },
  });

  const platformText = Platform.select({
    ios: 'iOS Specific Text',
    android: 'Android Specific Text',
    default: 'Default Text',
  });

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Platform Detection</Text>
      
      <View style={[styles.platformBox, platformStyles]}>
        <Text style={styles.platformText}>{platformText}</Text>
        {renderPlatformSpecific()}
      </View>

      <View style={styles.infoContainer}>
        <Text>Platform: {platformName}</Text>
        <Text>Version: {platformVersion}</Text>
        <Text>
          Is iOS: {Platform.OS === 'ios' ? 'Yes' : 'No'}
        </Text>
        <Text>
          Is Android: {Platform.OS === 'android' ? 'Yes' : 'No'}
        </Text>
      </View>

      {/* Conditional imports */}
      {Platform.OS === 'ios' && <IOSSpecificComponent />}
      {Platform.OS === 'android' && <AndroidSpecificComponent />}
    </View>
  );
}

// Platform-specific components
const IOSSpecificComponent = () => (
  <View style={styles.iosComponent}>
    <Text>iOS-only component with Cupertino styling</Text>
  </View>
);

const AndroidSpecificComponent = () => (
  <View style={styles.androidComponent}>
    <Text>Android-only component with Material styling</Text>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  platformBox: {
    padding: 20,
    borderRadius: 10,
    marginBottom: 20,
    alignItems: 'center',
  },
  platformText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  infoContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  iosComponent: {
    backgroundColor: '#E3F2FD',
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
  },
  androidComponent: {
    backgroundColor: '#E8F5E8',
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
  },
});
```

### Platform-Specific Files
```javascript
// Create platform-specific files:
// Button.ios.js
// Button.android.js
// Button.js (fallback)

// Button.ios.js
import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';

const Button = ({ title, onPress, style }) => (
  <TouchableOpacity style={[styles.button, style]} onPress={onPress}>
    <Text style={styles.text}>{title}</Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
  },
  text: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default Button;

// Button.android.js
import React from 'react';
import { TouchableNativeFeedback, Text, View, StyleSheet } from 'react-native';

const Button = ({ title, onPress, style }) => (
  <TouchableNativeFeedback onPress={onPress}>
    <View style={[styles.button, style]}>
      <Text style={styles.text}>{title}</Text>
    </View>
  </TouchableNativeFeedback>
);

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#4CAF50',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 4,
    alignItems: 'center',
    elevation: 2,
  },
  text: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500',
    textTransform: 'uppercase',
  },
});

export default Button;
```

## Native Modules and Bridges

### Creating a Native Module (Android)
```java
// android/app/src/main/java/com/yourapp/CalendarModule.java
package com.yourapp;

import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.Arguments;
import java.util.Map;
import java.util.HashMap;
import android.util.Log;

public class CalendarModule extends ReactContextBaseJavaModule {
    CalendarModule(ReactApplicationContext context) {
        super(context);
    }

    @Override
    public String getName() {
        return "CalendarModule";
    }

    @ReactMethod
    public void createCalendarEvent(String name, String location, Promise promise) {
        try {
            // Your native Android code here
            Log.d("CalendarModule", "Creating event: " + name + " at " + location);
            
            WritableMap event = Arguments.createMap();
            event.putString("name", name);
            event.putString("location", location);
            event.putString("id", "unique_event_id");
            
            promise.resolve(event);
        } catch (Exception e) {
            promise.reject("CREATE_EVENT_ERROR", e);
        }
    }

    @Override
    public Map<String, Object> getConstants() {
        final Map<String, Object> constants = new HashMap<>();
        constants.put("DEFAULT_EVENT_NAME", "New Event");
        return constants;
    }
}
```

### Creating a Native Module (iOS)
```objc
// ios/CalendarModule.h
#import <React/RCTBridgeModule.h>

@interface CalendarModule : NSObject <RCTBridgeModule>
@end

// ios/CalendarModule.m
#import "CalendarModule.h"
#import <React/RCTLog.h>

@implementation CalendarModule

RCT_EXPORT_MODULE();

RCT_EXPORT_METHOD(createCalendarEvent:(NSString *)name
                 location:(NSString *)location
                 resolver:(RCTPromiseResolveBlock)resolve
                 rejecter:(RCTPromiseRejectBlock)reject)
{
    @try {
        RCTLogInfo(@"Creating event %@ at %@", name, location);
        
        NSDictionary *event = @{
            @"name": name,
            @"location": location,
            @"id": @"unique_event_id"
        };
        
        resolve(event);
    }
    @catch (NSException *exception) {
        reject(@"create_event_error", @"Error creating event", nil);
    }
}

- (NSDictionary *)constantsToExport
{
    return @{ @"DEFAULT_EVENT_NAME": @"New Event" };
}

@end
```

### Using Native Modules in JavaScript
```javascript
// NativeModuleExample.js
import React, { useState } from 'react';
import { View, Text, Button, TextInput, Alert, NativeModules } from 'react-native';

const { CalendarModule } = NativeModules;

function NativeModuleExample() {
  const [eventName, setEventName] = useState('');
  const [eventLocation, setEventLocation] = useState('');

  const createEvent = async () => {
    try {
      const event = await CalendarModule.createCalendarEvent(
        eventName || CalendarModule.DEFAULT_EVENT_NAME,
        eventLocation
      );
      
      Alert.alert(
        'Success',
        `Event created: ${event.name} at ${event.location}`
      );
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Native Module Example</Text>
      
      <TextInput
        style={styles.input}
        placeholder="Event Name"
        value={eventName}
        onChangeText={setEventName}
      />
      
      <TextInput
        style={styles.input}
        placeholder="Event Location"
        value={eventLocation}
        onChangeText={setEventLocation}
      />
      
      <Button title="Create Event" onPress={createEvent} />
      
      <Text>Default event name: {CalendarModule.DEFAULT_EVENT_NAME}</Text>
    </View>
  );
}
```

## Device APIs

### Device Information
```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import DeviceInfo from 'react-native-device-info';

function DeviceInfoExample() {
  const [deviceInfo, setDeviceInfo] = useState({});

  useEffect(() => {
    getDeviceInfo();
  }, []);

  const getDeviceInfo = async () => {
    try {
      const info = {
        // Synchronous methods
        appName: DeviceInfo.getApplicationName(),
        buildNumber: DeviceInfo.getBuildNumber(),
        bundleId: DeviceInfo.getBundleId(),
        version: DeviceInfo.getVersion(),
        readableVersion: DeviceInfo.getReadableVersion(),
        deviceId: DeviceInfo.getDeviceId(),
        systemName: DeviceInfo.getSystemName(),
        systemVersion: DeviceInfo.getSystemVersion(),
        model: DeviceInfo.getModel(),
        brand: DeviceInfo.getBrand(),
        
        // Asynchronous methods
        uniqueId: await DeviceInfo.getUniqueId(),
        manufacturer: await DeviceInfo.getManufacturer(),
        deviceName: await DeviceInfo.getDeviceName(),
        isEmulator: await DeviceInfo.isEmulator(),
        hasNotch: DeviceInfo.hasNotch(),
        isTablet: DeviceInfo.isTablet(),
        
        // Battery and memory
        batteryLevel: await DeviceInfo.getBatteryLevel(),
        totalMemory: await DeviceInfo.getTotalMemory(),
        usedMemory: await DeviceInfo.getUsedMemory(),
        
        // Network info
        ipAddress: await DeviceInfo.getIpAddress(),
        macAddress: await DeviceInfo.getMacAddress(),
        
        // Storage
        totalDiskCapacity: await DeviceInfo.getTotalDiskCapacity(),
        freeDiskStorage: await DeviceInfo.getFreeDiskStorage(),
      };
      
      setDeviceInfo(info);
    } catch (error) {
      console.error('Error getting device info:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Device Information</Text>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>App Info</Text>
        <Text>Name: {deviceInfo.appName}</Text>
        <Text>Version: {deviceInfo.readableVersion}</Text>
        <Text>Build: {deviceInfo.buildNumber}</Text>
        <Text>Bundle ID: {deviceInfo.bundleId}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Device Info</Text>
        <Text>Name: {deviceInfo.deviceName}</Text>
        <Text>Model: {deviceInfo.model}</Text>
        <Text>Brand: {deviceInfo.brand}</Text>
        <Text>Manufacturer: {deviceInfo.manufacturer}</Text>
        <Text>System: {deviceInfo.systemName} {deviceInfo.systemVersion}</Text>
        <Text>Is Emulator: {deviceInfo.isEmulator ? 'Yes' : 'No'}</Text>
        <Text>Is Tablet: {deviceInfo.isTablet ? 'Yes' : 'No'}</Text>
        <Text>Has Notch: {deviceInfo.hasNotch ? 'Yes' : 'No'}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>System Resources</Text>
        <Text>Battery: {(deviceInfo.batteryLevel * 100).toFixed(0)}%</Text>
        <Text>Total Memory: {(deviceInfo.totalMemory / 1024 / 1024 / 1024).toFixed(2)} GB</Text>
        <Text>Used Memory: {(deviceInfo.usedMemory / 1024 / 1024).toFixed(0)} MB</Text>
        <Text>Total Storage: {(deviceInfo.totalDiskCapacity / 1024 / 1024 / 1024).toFixed(2)} GB</Text>
        <Text>Free Storage: {(deviceInfo.freeDiskStorage / 1024 / 1024 / 1024).toFixed(2)} GB</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Network</Text>
        <Text>IP Address: {deviceInfo.ipAddress}</Text>
        <Text>MAC Address: {deviceInfo.macAddress}</Text>
      </View>
    </View>
  );
}
```

### Haptic Feedback
```javascript
import React from 'react';
import { View, Button, StyleSheet } from 'react-native';
import { 
  trigger, 
  HapticFeedbackTypes,
  impactAsync, 
  ImpactFeedbackStyle,
  notificationAsync,
  NotificationFeedbackType,
  selectionAsync
} from 'expo-haptics';

function HapticFeedbackExample() {
  const triggerLight = () => {
    impactAsync(ImpactFeedbackStyle.Light);
  };

  const triggerMedium = () => {
    impactAsync(ImpactFeedbackStyle.Medium);
  };

  const triggerHeavy = () => {
    impactAsync(ImpactFeedbackStyle.Heavy);
  };

  const triggerSuccess = () => {
    notificationAsync(NotificationFeedbackType.Success);
  };

  const triggerWarning = () => {
    notificationAsync(NotificationFeedbackType.Warning);
  };

  const triggerError = () => {
    notificationAsync(NotificationFeedbackType.Error);
  };

  const triggerSelection = () => {
    selectionAsync();
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Haptic Feedback</Text>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Impact Feedback</Text>
        <Button title="Light Impact" onPress={triggerLight} />
        <Button title="Medium Impact" onPress={triggerMedium} />
        <Button title="Heavy Impact" onPress={triggerHeavy} />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notification Feedback</Text>
        <Button title="Success" onPress={triggerSuccess} />
        <Button title="Warning" onPress={triggerWarning} />
        <Button title="Error" onPress={triggerError} />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Selection Feedback</Text>
        <Button title="Selection" onPress={triggerSelection} />
      </View>
    </View>
  );
}
```

## Networking

### Advanced Networking with Fetch
```javascript
import React, { useState } from 'react';
import { View, Text, Button, FlatList, StyleSheet, Alert } from 'react-native';

function NetworkingExample() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Basic GET request
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('https://jsonplaceholder.typicode.com/posts');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setData(result.slice(0, 10)); // Limit to 10 items
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // POST request
  const createPost = async () => {
    setLoading(true);
    
    try {
      const response = await fetch('https://jsonplaceholder.typicode.com/posts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: 'New Post',
          body: 'This is a new post created from React Native',
          userId: 1,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const newPost = await response.json();
      setData(prevData => [newPost, ...prevData]);
      Alert.alert('Success', 'Post created successfully!');
    } catch (err) {
      Alert.alert('Error', err.message);
    } finally {
      setLoading(false);
    }
  };

  // Request with timeout and retry
  const fetchWithRetry = async (url, options = {}, retries = 3) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
      } catch (error) {
        if (i === retries - 1) throw error;
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1))); // Exponential backoff
      }
    }
  };

  const fetchWithRetryExample = async () => {
    setLoading(true);
    
    try {
      const result = await fetchWithRetry('https://jsonplaceholder.typicode.com/posts');
      setData(result.slice(0, 10));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderItem = ({ item }) => (
    <View style={styles.item}>
      <Text style={styles.itemTitle}>{item.title}</Text>
      <Text style={styles.itemBody} numberOfLines={2}>
        {item.body}
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Networking Example</Text>
      
      <View style={styles.buttonContainer}>
        <Button title="Fetch Data" onPress={fetchData} disabled={loading} />
        <Button title="Create Post" onPress={createPost} disabled={loading} />
        <Button title="Fetch with Retry" onPress={fetchWithRetryExample} disabled={loading} />
      </View>

      {loading && <Text style={styles.loading}>Loading...</Text>}
      {error && <Text style={styles.error}>Error: {error}</Text>}

      <FlatList
        data={data}
        renderItem={renderItem}
        keyExtractor={item => item.id.toString()}
        style={styles.list}
      />
    </View>
  );
}
```

### WebSocket Connection
```javascript
import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TextInput, Button, FlatList, StyleSheet } from 'react-native';

function WebSocketExample() {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket('wss://echo.websocket.org/');
      
      ws.onopen = () => {
        setIsConnected(true);
        setSocket(ws);
        reconnectAttempts.current = 0;
        addMessage('Connected to WebSocket server', 'system');
      };

      ws.onmessage = (event) => {
        addMessage(event.data, 'received');
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        setSocket(null);
        addMessage(`Connection closed: ${event.reason}`, 'system');
        
        // Auto-reconnect with exponential backoff
        if (reconnectAttempts.current < 5) {
          const timeout = Math.pow(2, reconnectAttempts.current) * 1000;
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttempts.current++;
            addMessage(`Reconnecting... (attempt ${reconnectAttempts.current})`, 'system');
            connectWebSocket();
          }, timeout);
        }
      };

      ws.onerror = (error) => {
        addMessage(`WebSocket error: ${error.message}`, 'error');
      };

    } catch (error) {
      addMessage(`Failed to connect: ${error.message}`, 'error');
    }
  };

  const disconnect = () => {
    if (socket) {
      socket.close();
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
  };

  const sendMessage = () => {
    if (socket && socket.readyState === WebSocket.OPEN && inputMessage.trim()) {
      socket.send(inputMessage);
      addMessage(inputMessage, 'sent');
      setInputMessage('');
    }
  };

  const addMessage = (text, type) => {
    const message = {
      id: Date.now().toString(),
      text,
      type,
      timestamp: new Date().toLocaleTimeString(),
    };
    setMessages(prev => [...prev, message]);
  };

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      disconnect();
    };
  }, []);

  const renderMessage = ({ item }) => (
    <View style={[styles.message, styles[item.type]]}>
      <Text style={styles.messageText}>{item.text}</Text>
      <Text style={styles.timestamp}>{item.timestamp}</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>WebSocket Example</Text>
      
      <View style={styles.statusContainer}>
        <Text style={[styles.status, isConnected ? styles.connected : styles.disconnected]}>
          Status: {isConnected ? 'Connected' : 'Disconnected'}
        </Text>
      </View>

      <FlatList
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        style={styles.messagesList}
      />

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={inputMessage}
          onChangeText={setInputMessage}
          placeholder="Enter message..."
          editable={isConnected}
        />
        <Button
          title="Send"
          onPress={sendMessage}
          disabled={!isConnected || !inputMessage.trim()}
        />
      </View>

      <View style={styles.buttonContainer}>
        <Button
          title={isConnected ? "Disconnect" : "Connect"}
          onPress={isConnected ? disconnect : connectWebSocket}
        />
      </View>
    </View>
  );
}
```

## Storage Solutions

### AsyncStorage Implementation
```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, FlatList, StyleSheet, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

function AsyncStorageExample() {
  const [key, setKey] = useState('');
  const [value, setValue] = useState('');
  const [storedItems, setStoredItems] = useState([]);

  useEffect(() => {
    loadAllItems();
  }, []);

  const storeData = async () => {
    if (!key || !value) {
      Alert.alert('Error', 'Please enter both key and value');
      return;
    }

    try {
      await AsyncStorage.setItem(key, value);
      Alert.alert('Success', 'Data stored successfully!');
      setKey('');
      setValue('');
      loadAllItems();
    } catch (error) {
      Alert.alert('Error', 'Failed to store data: ' + error.message);
    }
  };

  const storeObject = async () => {
    try {
      const user = {
        id: Date.now(),
        name: 'John Doe',
        email: 'john@example.com',
        preferences: {
          theme: 'dark',
          notifications: true,
        },
      };
      
      await AsyncStorage.setItem('user', JSON.stringify(user));
      Alert.alert('Success', 'Object stored successfully!');
      loadAllItems();
    } catch (error) {
      Alert.alert('Error', 'Failed to store object: ' + error.message);
    }
  };

  const getData = async (key) => {
    try {
      const value = await AsyncStorage.getItem(key);
      if (value !== null) {
        Alert.alert('Retrieved Data', `Key: ${key}\nValue: ${value}`);
      } else {
        Alert.alert('Not Found', 'No data found for this key');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to retrieve data: ' + error.message);
    }
  };

  const removeData = async (key) => {
    try {
      await AsyncStorage.removeItem(key);
      Alert.alert('Success', 'Data removed successfully!');
      loadAllItems();
    } catch (error) {
      Alert.alert('Error', 'Failed to remove data: ' + error.message);
    }
  };

  const clearAll = async () => {
    try {
      await AsyncStorage.clear();
      Alert.alert('Success', 'All data cleared!');
      setStoredItems([]);
    } catch (error) {
      Alert.alert('Error', 'Failed to clear data: ' + error.message);
    }
  };

  const loadAllItems = async () => {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const items = await AsyncStorage.multiGet(keys);
      
      const formattedItems = items.map(([key, value]) => ({
        key,
        value: value.length > 50 ? value.substring(0, 50) + '...' : value,
        fullValue: value,
      }));
      
      setStoredItems(formattedItems);
    } catch (error) {
      console.error('Failed to load items:', error);
    }
  };

  const renderStoredItem = ({ item }) => (
    <View style={styles.storedItem}>
      <View style={styles.itemContent}>
        <Text style={styles.itemKey}>{item.key}</Text>
        <Text style={styles.itemValue}>{item.value}</Text>
      </View>
      <View style={styles.itemActions}>
        <Button
          title="View"
          onPress={() => getData(item.key)}
          color="#007bff"
        />
        <Button
          title="Delete"
          onPress={() => removeData(item.key)}
          color="#dc3545"
        />
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AsyncStorage Example</Text>

      <View style={styles.inputSection}>
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
        <View style={styles.buttonRow}>
          <Button title="Store Data" onPress={storeData} />
          <Button title="Store Object" onPress={storeObject} />
        </View>
      </View>

      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Stored Items ({storedItems.length})</Text>
          <Button title="Clear All" onPress={clearAll} color="#dc3545" />
        </View>
        
        <FlatList
          data={storedItems}
          renderItem={renderStoredItem}
          keyExtractor={item => item.key}
          style={styles.list}
          showsVerticalScrollIndicator={false}
        />
      </View>
    </View>
  );
}
```

### SQLite Database
```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, FlatList, StyleSheet, Alert } from 'react-native';
import * as SQLite from 'expo-sqlite';

function SQLiteExample() {
  const [db, setDb] = useState(null);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [users, setUsers] = useState([]);

  useEffect(() => {
    initDatabase();
  }, []);

  const initDatabase = async () => {
    try {
      const database = await SQLite.openDatabaseAsync('users.db');
      setDb(database);
      
      // Create table if it doesn't exist
      await database.execAsync(`
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          email TEXT UNIQUE NOT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
      `);
      
      loadUsers(database);
    } catch (error) {
      Alert.alert('Database Error', error.message);
    }
  };

  const addUser = async () => {
    if (!name || !email) {
      Alert.alert('Error', 'Please enter both name and email');
      return;
    }

    try {
      await db.runAsync(
        'INSERT INTO users (name, email) VALUES (?, ?)',
        [name, email]
      );
      
      Alert.alert('Success', 'User added successfully!');
      setName('');
      setEmail('');
      loadUsers(db);
    } catch (error) {
      Alert.alert('Error', 'Failed to add user: ' + error.message);
    }
  };

  const loadUsers = async (database = db) => {
    try {
      const result = await database.getAllAsync('SELECT * FROM users ORDER BY created_at DESC');
      setUsers(result);
    } catch (error) {
      console.error('Failed to load users:', error);
    }
  };

  const deleteUser = async (id) => {
    try {
      await db.runAsync('DELETE FROM users WHERE id = ?', [id]);
      Alert.alert('Success', 'User deleted successfully!');
      loadUsers(db);
    } catch (error) {
      Alert.alert('Error', 'Failed to delete user: ' + error.message);
    }
  };

  const updateUser = async (id, newName, newEmail) => {
    try {
      await db.runAsync(
        'UPDATE users SET name = ?, email = ? WHERE id = ?',
        [newName, newEmail, id]
      );
      Alert.alert('Success', 'User updated successfully!');
      loadUsers(db);
    } catch (error) {
      Alert.alert('Error', 'Failed to update user: ' + error.message);
    }
  };

  const clearDatabase = async () => {
    try {
      await db.runAsync('DELETE FROM users');
      Alert.alert('Success', 'All users cleared!');
      loadUsers(db);
    } catch (error) {
      Alert.alert('Error', 'Failed to clear database: ' + error.message);
    }
  };

  const renderUser = ({ item }) => (
    <View style={styles.userItem}>
      <View style={styles.userInfo}>
        <Text style={styles.userName}>{item.name}</Text>
        <Text style={styles.userEmail}>{item.email}</Text>
        <Text style={styles.userDate}>
          Created: {new Date(item.created_at).toLocaleDateString()}
        </Text>
      </View>
      <View style={styles.userActions}>
        <Button
          title="Edit"
          onPress={() => {
            Alert.prompt(
              'Edit User',
              'Enter new name:',
              (newName) => {
                if (newName) {
                  Alert.prompt(
                    'Edit User',
                    'Enter new email:',
                    (newEmail) => {
                      if (newEmail) {
                        updateUser(item.id, newName, newEmail);
                      }
                    },
                    'plain-text',
                    item.email
                  );
                }
              },
              'plain-text',
              item.name
            );
          }}
          color="#007bff"
        />
        <Button
          title="Delete"
          onPress={() => deleteUser(item.id)}
          color="#dc3545"
        />
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>SQLite Database Example</Text>

      <View style={styles.inputSection}>
        <TextInput
          style={styles.input}
          placeholder="Enter name"
          value={name}
          onChangeText={setName}
        />
        <TextInput
          style={styles.input}
          placeholder="Enter email"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
        />
        <Button title="Add User" onPress={addUser} />
      </View>

      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Users ({users.length})</Text>
          <Button title="Clear All" onPress={clearDatabase} color="#dc3545" />
        </View>
        
        <FlatList
          data={users}
          renderItem={renderUser}
          keyExtractor={item => item.id.toString()}
          style={styles.list}
        />
      </View>
    </View>
  );
}
```

## Background Tasks

### Background App Refresh
```javascript
import React, { useEffect, useState } from 'react';
import { View, Text, Button, StyleSheet, AppState } from 'react-native';
import BackgroundTimer from 'react-native-background-timer';
import * as TaskManager from 'expo-task-manager';
import * as BackgroundFetch from 'expo-background-fetch';

const BACKGROUND_FETCH_TASK = 'background-fetch';

// Define the background task
TaskManager.defineTask(BACKGROUND_FETCH_TASK, async () => {
  const now = Date.now();
  console.log(`Got background fetch call at date: ${new Date(now).toISOString()}`);
  
  // Perform background work here
  try {
    // Example: sync data, check for updates, etc.
    const response = await fetch('https://api.example.com/sync');
    const data = await response.json();
    
    // Store the result
    await AsyncStorage.setItem('lastBackgroundSync', now.toString());
    
    return BackgroundFetch.BackgroundFetchResult.NewData;
  } catch (error) {
    return BackgroundFetch.BackgroundFetchResult.Failed;
  }
});

function BackgroundTaskExample() {
  const [appState, setAppState] = useState(AppState.currentState);
  const [backgroundTime, setBackgroundTime] = useState(0);
  const [isBackgroundFetchEnabled, setIsBackgroundFetchEnabled] = useState(false);

  useEffect(() => {
    registerBackgroundTask();
    
    const handleAppStateChange = (nextAppState) => {
      if (appState.match(/inactive|background/) && nextAppState === 'active') {
        console.log('App has come to the foreground!');
        stopBackgroundTimer();
      } else if (nextAppState.match(/inactive|background/)) {
        console.log('App has gone to the background!');
        startBackgroundTimer();
      }
      setAppState(nextAppState);
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);
    
    return () => {
      subscription?.remove();
      stopBackgroundTimer();
    };
  }, [appState]);

  const registerBackgroundTask = async () => {
    try {
      const status = await BackgroundFetch.getStatusAsync();
      console.log('Background fetch status:', status);
      
      if (status === BackgroundFetch.BackgroundFetchStatus.Available) {
        await BackgroundFetch.registerTaskAsync(BACKGROUND_FETCH_TASK, {
          minimumInterval: 15000, // 15 seconds (minimum for testing)
          stopOnTerminate: false, // Continue after app termination
          startOnBoot: true, // Start after device restart
        });
        setIsBackgroundFetchEnabled(true);
        console.log('Background fetch task registered');
      }
    } catch (error) {
      console.error('Failed to register background task:', error);
    }
  };

  const unregisterBackgroundTask = async () => {
    try {
      await BackgroundFetch.unregisterTaskAsync(BACKGROUND_FETCH_TASK);
      setIsBackgroundFetchEnabled(false);
      console.log('Background fetch task unregistered');
    } catch (error) {
      console.error('Failed to unregister background task:', error);
    }
  };

  const startBackgroundTimer = () => {
    BackgroundTimer.start();
    
    const timer = setInterval(() => {
      setBackgroundTime(prev => prev + 1);
      console.log('Background timer tick:', Date.now());
    }, 1000);

    BackgroundTimer.backgroundJob = timer;
  };

  const stopBackgroundTimer = () => {
    if (BackgroundTimer.backgroundJob) {
      clearInterval(BackgroundTimer.backgroundJob);
      BackgroundTimer.backgroundJob = null;
    }
    BackgroundTimer.stop();
  };

  const triggerBackgroundFetch = async () => {
    try {
      const result = await BackgroundFetch.BackgroundFetchResult.NewData;
      console.log('Manual background fetch result:', result);
    } catch (error) {
      console.error('Manual background fetch failed:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Background Tasks</Text>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>App State</Text>
        <Text>Current State: {appState}</Text>
        <Text>Background Time: {backgroundTime} seconds</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Background Fetch</Text>
        <Text>Status: {isBackgroundFetchEnabled ? 'Enabled' : 'Disabled'}</Text>
        
        <View style={styles.buttonContainer}>
          <Button
            title="Register Task"
            onPress={registerBackgroundTask}
            disabled={isBackgroundFetchEnabled}
          />
          <Button
            title="Unregister Task"
            onPress={unregisterBackgroundTask}
            disabled={!isBackgroundFetchEnabled}
          />
          <Button
            title="Trigger Fetch"
            onPress={triggerBackgroundFetch}
            disabled={!isBackgroundFetchEnabled}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Background Timer</Text>
        <View style={styles.buttonContainer}>
          <Button
            title="Start Timer"
            onPress={startBackgroundTimer}
          />
          <Button
            title="Stop Timer"
            onPress={stopBackgroundTimer}
          />
        </View>
      </View>
    </View>
  );
}
```

## Linking and Deep Links

### URL Scheme Handling
```javascript
import React, { useEffect, useState } from 'react';
import { View, Text, Button, Linking, StyleSheet, Alert } from 'react-native';

function LinkingExample() {
  const [initialUrl, setInitialUrl] = useState(null);
  const [lastUrl, setLastUrl] = useState(null);

  useEffect(() => {
    // Get the initial URL if the app was opened via a deep link
    Linking.getInitialURL().then((url) => {
      if (url) {
        setInitialUrl(url);
        handleDeepLink(url);
      }
    });

    // Listen for incoming links when the app is already open
    const subscription = Linking.addEventListener('url', ({ url }) => {
      setLastUrl(url);
      handleDeepLink(url);
    });

    return () => {
      subscription?.remove();
    };
  }, []);

  const handleDeepLink = (url) => {
    console.log('Deep link received:', url);
    
    // Parse the URL and extract parameters
    const route = parseDeepLink(url);
    
    if (route) {
      Alert.alert(
        'Deep Link Received',
        `Screen: ${route.screen}\nParams: ${JSON.stringify(route.params)}`
      );
      
      // Navigate to the appropriate screen
      // navigation.navigate(route.screen, route.params);
    }
  };

  const parseDeepLink = (url) => {
    // Example URL schemes:
    // myapp://profile/123
    // myapp://product/456?category=electronics
    
    const urlParts = url.replace('myapp://', '').split('?');
    const pathParts = urlParts[0].split('/');
    const queryString = urlParts[1];
    
    const screen = pathParts[0];
    const id = pathParts[1];
    
    let params = { id };
    
    // Parse query parameters
    if (queryString) {
      const queryParams = new URLSearchParams(queryString);
      queryParams.forEach((value, key) => {
        params[key] = value;
      });
    }
    
    return { screen, params };
  };

  const openExternalUrl = async (url) => {
    try {
      const supported = await Linking.canOpenURL(url);
      
      if (supported) {
        await Linking.openURL(url);
      } else {
        Alert.alert('Error', `Can't open URL: ${url}`);
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };

  const makePhoneCall = async (phoneNumber) => {
    const url = `tel:${phoneNumber}`;
    const supported = await Linking.canOpenURL(url);
    
    if (supported) {
      await Linking.openURL(url);
    } else {
      Alert.alert('Error', 'Phone calls are not supported on this device');
    }
  };

  const sendEmail = async (email, subject = '', body = '') => {
    const url = `mailto:${email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    const supported = await Linking.canOpenURL(url);
    
    if (supported) {
      await Linking.openURL(url);
    } else {
      Alert.alert('Error', 'Email is not supported on this device');
    }
  };

  const openMaps = async (address) => {
    const encodedAddress = encodeURIComponent(address);
    const url = Platform.select({
      ios: `maps:q=${encodedAddress}`,
      android: `geo:0,0?q=${encodedAddress}`,
    });
    
    const supported = await Linking.canOpenURL(url);
    
    if (supported) {
      await Linking.openURL(url);
    } else {
      // Fallback to web maps
      await Linking.openURL(`https://maps.google.com/maps?q=${encodedAddress}`);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Linking & Deep Links</Text>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Deep Link Info</Text>
        <Text>Initial URL: {initialUrl || 'None'}</Text>
        <Text>Last URL: {lastUrl || 'None'}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Open External URLs</Text>
        <View style={styles.buttonContainer}>
          <Button
            title="Open Website"
            onPress={() => openExternalUrl('https://www.google.com')}
          />
          <Button
            title="Open App Store"
            onPress={() => openExternalUrl('https://apps.apple.com/app/id123456789')}
          />
          <Button
            title="Open Play Store"
            onPress={() => openExternalUrl('market://details?id=com.example.app')}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Native Actions</Text>
        <View style={styles.buttonContainer}>
          <Button
            title="Make Phone Call"
            onPress={() => makePhoneCall('+1234567890')}
          />
          <Button
            title="Send Email"
            onPress={() => sendEmail('test@example.com', 'Hello', 'This is a test email')}
          />
          <Button
            title="Open Maps"
            onPress={() => openMaps('1600 Amphitheatre Parkway, Mountain View, CA')}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Social Media</Text>
        <View style={styles.buttonContainer}>
          <Button
            title="Open Twitter"
            onPress={() => openExternalUrl('twitter://user?screen_name=username')}
          />
          <Button
            title="Open Instagram"
            onPress={() => openExternalUrl('instagram://user?username=username')}
          />
          <Button
            title="Open WhatsApp"
            onPress={() => openExternalUrl('whatsapp://send?text=Hello')}
          />
        </View>
      </View>
    </View>
  );
}
```

## Permissions

### Permission Management
```javascript
import React, { useState, useEffect } from 'react';
import { View, Text, Button, StyleSheet, Alert, Platform } from 'react-native';
import { 
  request, 
  requestMultiple,
  check, 
  checkMultiple,
  openSettings,
  PERMISSIONS, 
  RESULTS 
} from 'react-native-permissions';

function PermissionsExample() {
  const [permissions, setPermissions] = useState({});

  const PERMISSION_LIST = Platform.select({
    ios: [
      PERMISSIONS.IOS.CAMERA,
      PERMISSIONS.IOS.PHOTO_LIBRARY,
      PERMISSIONS.IOS.MICROPHONE,
      PERMISSIONS.IOS.LOCATION_WHEN_IN_USE,
      PERMISSIONS.IOS.CONTACTS,
      PERMISSIONS.IOS.NOTIFICATIONS,
    ],
    android: [
      PERMISSIONS.ANDROID.CAMERA,
      PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE,
      PERMISSIONS.ANDROID.WRITE_EXTERNAL_STORAGE,
      PERMISSIONS.ANDROID.RECORD_AUDIO,
      PERMISSIONS.ANDROID.ACCESS_FINE_LOCATION,
      PERMISSIONS.ANDROID.READ_CONTACTS,
    ],
  });

  useEffect(() => {
    checkAllPermissions();
  }, []);

  const checkAllPermissions = async () => {
    try {
      const statuses = await checkMultiple(PERMISSION_LIST);
      setPermissions(statuses);
    } catch (error) {
      console.error('Error checking permissions:', error);
    }
  };

  const requestPermission = async (permission) => {
    try {
      const result = await request(permission);
      
      setPermissions(prev => ({
        ...prev,
        [permission]: result,
      }));

      switch (result) {
        case RESULTS.UNAVAILABLE:
          Alert.alert('Permission Unavailable', 'This feature is not available on this device');
          break;
        case RESULTS.DENIED:
          Alert.alert('Permission Denied', 'Permission was denied');
          break;
        case RESULTS.LIMITED:
          Alert.alert('Permission Limited', 'Permission was granted with limitations');
          break;
        case RESULTS.GRANTED:
          Alert.alert('Permission Granted', 'Permission was granted successfully');
          break;
        case RESULTS.BLOCKED:
          Alert.alert(
            'Permission Blocked',
            'Permission is blocked. Please enable it in settings.',
            [
              { text: 'Cancel', style: 'cancel' },
              { text: 'Open Settings', onPress: openSettings },
            ]
          );
          break;
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };

  const requestAllPermissions = async () => {
    try {
      const statuses = await requestMultiple(PERMISSION_LIST);
      setPermissions(statuses);
      
      const deniedPermissions = Object.keys(statuses).filter(
        permission => statuses[permission] === RESULTS.DENIED || statuses[permission] === RESULTS.BLOCKED
      );
      
      if (deniedPermissions.length > 0) {
        Alert.alert(
          'Some Permissions Denied',
          `The following permissions were denied: ${deniedPermissions.map(p => getPermissionName(p)).join(', ')}`
        );
      } else {
        Alert.alert('Success', 'All permissions granted!');
      }
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };

  const getPermissionName = (permission) => {
    const names = {
      [PERMISSIONS.IOS.CAMERA]: 'Camera',
      [PERMISSIONS.IOS.PHOTO_LIBRARY]: 'Photo Library',
      [PERMISSIONS.IOS.MICROPHONE]: 'Microphone',
      [PERMISSIONS.IOS.LOCATION_WHEN_IN_USE]: 'Location',
      [PERMISSIONS.IOS.CONTACTS]: 'Contacts',
      [PERMISSIONS.IOS.NOTIFICATIONS]: 'Notifications',
      [PERMISSIONS.ANDROID.CAMERA]: 'Camera',
      [PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE]: 'Read Storage',
      [PERMISSIONS.ANDROID.WRITE_EXTERNAL_STORAGE]: 'Write Storage',
      [PERMISSIONS.ANDROID.RECORD_AUDIO]: 'Record Audio',
      [PERMISSIONS.ANDROID.ACCESS_FINE_LOCATION]: 'Location',
      [PERMISSIONS.ANDROID.READ_CONTACTS]: 'Read Contacts',
    };
    return names[permission] || permission;
  };

  const getStatusColor = (status) => {
    switch (status) {
      case RESULTS.GRANTED:
        return '#28a745';
      case RESULTS.LIMITED:
        return '#ffc107';
      case RESULTS.DENIED:
        return '#dc3545';
      case RESULTS.BLOCKED:
        return '#6c757d';
      case RESULTS.UNAVAILABLE:
        return '#6c757d';
      default:
        return '#6c757d';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case RESULTS.GRANTED:
        return 'Granted';
      case RESULTS.LIMITED:
        return 'Limited';
      case RESULTS.DENIED:
        return 'Denied';
      case RESULTS.BLOCKED:
        return 'Blocked';
      case RESULTS.UNAVAILABLE:
        return 'Unavailable';
      default:
        return 'Unknown';
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Permissions Management</Text>
      
      <View style={styles.buttonContainer}>
        <Button title="Check All Permissions" onPress={checkAllPermissions} />
        <Button title="Request All Permissions" onPress={requestAllPermissions} />
        <Button title="Open Settings" onPress={openSettings} />
      </View>

      <View style={styles.permissionsList}>
        <Text style={styles.sectionTitle}>Permission Status:</Text>
        
        {PERMISSION_LIST.map(permission => {
          const status = permissions[permission];
          const name = getPermissionName(permission);
          const color = getStatusColor(status);
          const statusText = getStatusText(status);
          
          return (
            <View key={permission} style={styles.permissionItem}>
              <View style={styles.permissionInfo}>
                <Text style={styles.permissionName}>{name}</Text>
                <Text style={[styles.permissionStatus, { color }]}>
                  {statusText}
                </Text>
              </View>
              <Button
                title="Request"
                onPress={() => requestPermission(permission)}
                disabled={status === RESULTS.GRANTED}
                color={status === RESULTS.GRANTED ? '#28a745' : '#007bff'}
              />
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
    flexWrap: 'wrap',
    gap: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  permissionsList: {
    flex: 1,
  },
  permissionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 15,
    marginBottom: 8,
    borderRadius: 8,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  permissionInfo: {
    flex: 1,
  },
  permissionName: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 4,
  },
  permissionStatus: {
    fontSize: 14,
    fontWeight: 'bold',
  },
});

export default PermissionsExample;
```

---

*Continue to: [15-react-native-performance.md](./15-react-native-performance.md)*
