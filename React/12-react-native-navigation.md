# React Native Navigation

## Table of Contents
- [React Navigation Setup](#react-navigation-setup)
- [Stack Navigation](#stack-navigation)
- [Tab Navigation](#tab-navigation)
- [Drawer Navigation](#drawer-navigation)
- [Nested Navigation](#nested-navigation)
- [Navigation Patterns](#navigation-patterns)
- [Deep Linking](#deep-linking)
- [Navigation Guards](#navigation-guards)

## React Navigation Setup

### Installation and Configuration
```bash
# Install React Navigation
npm install @react-navigation/native

# Install dependencies for Expo
npx expo install react-native-screens react-native-safe-area-context

# For bare React Native projects
npm install react-native-screens react-native-safe-area-context
cd ios && pod install

# Install navigators
npm install @react-navigation/stack
npm install @react-navigation/bottom-tabs
npm install @react-navigation/drawer
npm install @react-navigation/material-top-tabs
```

### Basic Navigation Container Setup
```javascript
// App.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { enableScreens } from 'react-native-screens';

// Enable screens for better performance
enableScreens();

import HomeScreen from './screens/HomeScreen';
import DetailsScreen from './screens/DetailsScreen';
import ProfileScreen from './screens/ProfileScreen';

const Stack = createStackNavigator();

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#007bff',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen 
          name="Home" 
          component={HomeScreen}
          options={{ title: 'Welcome Home' }}
        />
        <Stack.Screen 
          name="Details" 
          component={DetailsScreen}
          options={({ route }) => ({ 
            title: route.params?.title || 'Details' 
          })}
        />
        <Stack.Screen 
          name="Profile" 
          component={ProfileScreen}
          options={{
            headerRight: () => (
              <Button
                onPress={() => alert('Settings')}
                title="Settings"
                color="#fff"
              />
            ),
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;
```

## Stack Navigation

### Basic Stack Navigation
```javascript
// screens/HomeScreen.js
import React from 'react';
import { View, Text, Button, StyleSheet, FlatList } from 'react-native';

function HomeScreen({ navigation }) {
  const navigationOptions = [
    { id: '1', title: 'Go to Details', screen: 'Details', params: { itemId: 86, title: 'Product Details' } },
    { id: '2', title: 'Go to Profile', screen: 'Profile' },
    { id: '3', title: 'Push Details', screen: 'Details', action: 'push' },
    { id: '4', title: 'Replace with Profile', screen: 'Profile', action: 'replace' },
  ];

  const handleNavigation = (item) => {
    switch (item.action) {
      case 'push':
        navigation.push(item.screen, item.params);
        break;
      case 'replace':
        navigation.replace(item.screen, item.params);
        break;
      default:
        navigation.navigate(item.screen, item.params);
    }
  };

  const renderNavigationItem = ({ item }) => (
    <View style={styles.buttonContainer}>
      <Button
        title={item.title}
        onPress={() => handleNavigation(item)}
      />
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Home Screen</Text>
      <Text style={styles.subtitle}>Navigation Options:</Text>
      
      <FlatList
        data={navigationOptions}
        renderItem={renderNavigationItem}
        keyExtractor={item => item.id}
        style={styles.list}
      />

      <View style={styles.infoContainer}>
        <Text style={styles.infoTitle}>Navigation State:</Text>
        <Text style={styles.infoText}>Current Route: {navigation.getState().routeNames[navigation.getState().index]}</Text>
        <Text style={styles.infoText}>Can Go Back: {navigation.canGoBack() ? 'Yes' : 'No'}</Text>
      </View>
    </View>
  );
}

// screens/DetailsScreen.js
import React, { useLayoutEffect } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

function DetailsScreen({ route, navigation }) {
  const { itemId, title } = route.params || {};

  useLayoutEffect(() => {
    navigation.setOptions({
      title: title || `Details #${itemId}`,
      headerRight: () => (
        <Button
          onPress={() => navigation.navigate('Home')}
          title="Home"
          color="#fff"
        />
      ),
    });
  }, [navigation, title, itemId]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Details Screen</Text>
      
      {itemId && (
        <Text style={styles.subtitle}>Item ID: {itemId}</Text>
      )}

      <View style={styles.buttonContainer}>
        <Button
          title="Go to Details... again"
          onPress={() => navigation.push('Details', {
            itemId: Math.floor(Math.random() * 100),
          })}
        />
      </View>

      <View style={styles.buttonContainer}>
        <Button
          title="Go to Home"
          onPress={() => navigation.navigate('Home')}
        />
      </View>

      <View style={styles.buttonContainer}>
        <Button
          title="Go back"
          onPress={() => navigation.goBack()}
        />
      </View>

      <View style={styles.buttonContainer}>
        <Button
          title="Pop to top"
          onPress={() => navigation.popToTop()}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f8f9fa',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 20,
    color: '#666',
  },
  list: {
    flex: 1,
  },
  buttonContainer: {
    marginVertical: 8,
  },
  infoContainer: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 8,
    marginTop: 20,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
});
```

### Advanced Stack Configuration
```javascript
// navigation/StackNavigator.js
import React from 'react';
import { createStackNavigator, TransitionPresets } from '@react-navigation/stack';
import { Platform } from 'react-native';

const Stack = createStackNavigator();

function AdvancedStackNavigator() {
  return (
    <Stack.Navigator
      initialRouteName="Home"
      screenOptions={{
        // Header styling
        headerStyle: {
          backgroundColor: '#007bff',
          elevation: 0, // Android shadow
          shadowOpacity: 0, // iOS shadow
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: 18,
        },
        headerTitleAlign: 'center',
        
        // Transitions
        ...TransitionPresets.SlideFromRightIOS,
        
        // Gestures
        gestureEnabled: true,
        gestureDirection: 'horizontal',
        
        // Modal presentation
        cardStyle: { backgroundColor: 'transparent' },
      }}
    >
      <Stack.Screen 
        name="Home" 
        component={HomeScreen}
        options={{
          title: 'Home',
          headerLeft: null, // Disable back button
        }}
      />
      
      <Stack.Screen 
        name="Details" 
        component={DetailsScreen}
        options={({ route, navigation }) => ({
          title: route.params?.title || 'Details',
          
          // Custom header
          headerTitle: (props) => (
            <CustomHeaderTitle {...props} />
          ),
          
          // Animation override
          ...TransitionPresets.ModalSlideFromBottomIOS,
          
          // Custom back behavior
          headerLeft: () => (
            <HeaderBackButton
              onPress={() => {
                // Custom back action
                navigation.goBack();
              }}
            />
          ),
        })}
      />

      {/* Modal screen */}
      <Stack.Screen
        name="Modal"
        component={ModalScreen}
        options={{
          presentation: 'modal',
          ...TransitionPresets.ModalSlideFromBottomIOS,
        }}
      />

      {/* Transparent modal */}
      <Stack.Screen
        name="TransparentModal"
        component={TransparentModalScreen}
        options={{
          presentation: 'transparentModal',
          cardStyle: { backgroundColor: 'rgba(0, 0, 0, 0.5)' },
          cardOverlayEnabled: true,
          gestureEnabled: true,
          ...TransitionPresets.ModalSlideFromBottomIOS,
        }}
      />
    </Stack.Navigator>
  );
}

// Custom Header Component
function CustomHeaderTitle({ children, style }) {
  return (
    <View style={[{ flexDirection: 'row', alignItems: 'center' }, style]}>
      <Text style={{ color: 'white', fontSize: 18, fontWeight: 'bold' }}>
        {children}
      </Text>
    </View>
  );
}
```

## Tab Navigation

### Bottom Tab Navigator
```javascript
// navigation/TabNavigator.js
import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, Text, Platform } from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';

import HomeScreen from '../screens/HomeScreen';
import SearchScreen from '../screens/SearchScreen';
import FavoritesScreen from '../screens/FavoritesScreen';
import ProfileScreen from '../screens/ProfileScreen';

const Tab = createBottomTabNavigator();

function TabNavigator() {
  return (
    <Tab.Navigator
      initialRouteName="Home"
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          switch (route.name) {
            case 'Home':
              iconName = focused ? 'home' : 'home-outline';
              break;
            case 'Search':
              iconName = focused ? 'search' : 'search-outline';
              break;
            case 'Favorites':
              iconName = focused ? 'heart' : 'heart-outline';
              break;
            case 'Profile':
              iconName = focused ? 'person' : 'person-outline';
              break;
            default:
              iconName = 'circle';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        
        // Tab bar styling
        tabBarActiveTintColor: '#007bff',
        tabBarInactiveTintColor: 'gray',
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '600',
        },
        tabBarStyle: {
          backgroundColor: 'white',
          borderTopWidth: 1,
          borderTopColor: '#e0e0e0',
          height: Platform.OS === 'ios' ? 90 : 60,
          paddingBottom: Platform.OS === 'ios' ? 20 : 10,
          paddingTop: 10,
        },
        
        // Header styling
        headerStyle: {
          backgroundColor: '#007bff',
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      })}
    >
      <Tab.Screen 
        name="Home" 
        component={HomeScreen}
        options={{
          title: 'Home',
          tabBarBadge: 3, // Show badge
        }}
      />
      
      <Tab.Screen 
        name="Search" 
        component={SearchScreen}
        options={{
          title: 'Search',
          // Dynamic badge
          tabBarBadge: ({ focused }) => focused ? null : '!',
        }}
      />
      
      <Tab.Screen 
        name="Favorites" 
        component={FavoritesScreen}
        options={({ navigation, route }) => ({
          title: 'Favorites',
          // Custom tab bar button
          tabBarButton: (props) => (
            <CustomTabButton {...props} />
          ),
        })}
      />
      
      <Tab.Screen 
        name="Profile" 
        component={ProfileScreen}
        options={{
          title: 'Profile',
          // Hide tab bar on this screen
          tabBarStyle: { display: 'none' },
        }}
      />
    </Tab.Navigator>
  );
}

// Custom Tab Button
function CustomTabButton({ children, onPress, accessibilityState }) {
  const focused = accessibilityState?.selected;
  
  return (
    <View style={{
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: focused ? '#007bff' : 'transparent',
      borderRadius: focused ? 25 : 0,
      margin: focused ? 5 : 0,
    }}>
      <TouchableOpacity
        style={{
          flex: 1,
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%',
        }}
        onPress={onPress}
      >
        {children}
      </TouchableOpacity>
    </View>
  );
}
```

### Material Top Tabs
```javascript
// navigation/TopTabNavigator.js
import React from 'react';
import { createMaterialTopTabNavigator } from '@react-navigation/material-top-tabs';
import { View, Text, Dimensions } from 'react-native';

const Tab = createMaterialTopTabNavigator();
const { width } = Dimensions.get('window');

function TopTabNavigator() {
  return (
    <Tab.Navigator
      initialRouteName="Recent"
      screenOptions={{
        tabBarActiveTintColor: '#007bff',
        tabBarInactiveTintColor: 'gray',
        tabBarIndicatorStyle: {
          backgroundColor: '#007bff',
          height: 3,
        },
        tabBarLabelStyle: {
          fontSize: 14,
          fontWeight: 'bold',
          textTransform: 'none',
        },
        tabBarStyle: {
          backgroundColor: 'white',
          elevation: 0,
          shadowOpacity: 0,
        },
        tabBarScrollEnabled: true,
        tabBarItemStyle: {
          width: width / 3, // Fit 3 tabs on screen
        },
        // Swipe enabled
        swipeEnabled: true,
        // Lazy loading
        lazy: true,
      }}
    >
      <Tab.Screen
        name="Recent"
        component={RecentScreen}
        options={{
          tabBarLabel: 'Recent',
          tabBarIcon: ({ color }) => (
            <Icon name="time-outline" color={color} size={20} />
          ),
        }}
      />
      
      <Tab.Screen
        name="Popular"
        component={PopularScreen}
        options={{
          tabBarLabel: 'Popular',
          tabBarIcon: ({ color }) => (
            <Icon name="trending-up-outline" color={color} size={20} />
          ),
        }}
      />
      
      <Tab.Screen
        name="Trending"
        component={TrendingScreen}
        options={{
          tabBarLabel: 'Trending',
          tabBarIcon: ({ color }) => (
            <Icon name="flame-outline" color={color} size={20} />
          ),
        }}
      />
      
      <Tab.Screen
        name="Featured"
        component={FeaturedScreen}
        options={{
          tabBarLabel: 'Featured',
          tabBarIcon: ({ color }) => (
            <Icon name="star-outline" color={color} size={20} />
          ),
        }}
      />
    </Tab.Navigator>
  );
}
```

## Drawer Navigation

### Drawer Navigator Setup
```javascript
// navigation/DrawerNavigator.js
import React from 'react';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { View, Text, Image, StyleSheet } from 'react-native';

import HomeScreen from '../screens/HomeScreen';
import ProfileScreen from '../screens/ProfileScreen';
import SettingsScreen from '../screens/SettingsScreen';

const Drawer = createDrawerNavigator();

function DrawerNavigator() {
  return (
    <Drawer.Navigator
      initialRouteName="Home"
      drawerContent={(props) => <CustomDrawerContent {...props} />}
      screenOptions={{
        drawerStyle: {
          backgroundColor: '#f8f9fa',
          width: 280,
        },
        drawerActiveTintColor: '#007bff',
        drawerInactiveTintColor: '#666',
        drawerLabelStyle: {
          fontSize: 16,
          fontWeight: '500',
        },
        drawerItemStyle: {
          marginVertical: 2,
        },
        // Header styling
        headerStyle: {
          backgroundColor: '#007bff',
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      }}
    >
      <Drawer.Screen
        name="Home"
        component={HomeScreen}
        options={{
          drawerLabel: 'Home',
          drawerIcon: ({ focused, size, color }) => (
            <Icon name="home-outline" size={size} color={color} />
          ),
        }}
      />
      
      <Drawer.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          drawerLabel: 'Profile',
          drawerIcon: ({ focused, size, color }) => (
            <Icon name="person-outline" size={size} color={color} />
          ),
          // Badge
          drawerBadge: () => (
            <View style={styles.badge}>
              <Text style={styles.badgeText}>3</Text>
            </View>
          ),
        }}
      />
      
      <Drawer.Screen
        name="Settings"
        component={SettingsScreen}
        options={{
          drawerLabel: 'Settings',
          drawerIcon: ({ focused, size, color }) => (
            <Icon name="settings-outline" size={size} color={color} />
          ),
        }}
      />
    </Drawer.Navigator>
  );
}

// Custom Drawer Content
function CustomDrawerContent({ navigation, state, descriptors }) {
  return (
    <DrawerContentScrollView {...props}>
      {/* Header */}
      <View style={styles.drawerHeader}>
        <Image
          source={{ uri: 'https://via.placeholder.com/80' }}
          style={styles.userAvatar}
        />
        <Text style={styles.userName}>John Doe</Text>
        <Text style={styles.userEmail}>john@example.com</Text>
      </View>

      {/* Navigation Items */}
      <View style={styles.drawerSection}>
        {state.routes.map((route, index) => {
          const { options } = descriptors[route.key];
          const label = options.drawerLabel || route.name;
          const isFocused = state.index === index;

          const onPress = () => {
            const event = navigation.emit({
              type: 'drawerItemPress',
              target: route.key,
              canPreventDefault: true,
            });

            if (!isFocused && !event.defaultPrevented) {
              navigation.navigate(route.name);
            }
          };

          return (
            <DrawerItem
              key={route.key}
              label={label}
              focused={isFocused}
              onPress={onPress}
              icon={options.drawerIcon}
              style={[
                styles.drawerItem,
                isFocused && styles.drawerItemFocused
              ]}
              labelStyle={[
                styles.drawerLabel,
                isFocused && styles.drawerLabelFocused
              ]}
            />
          );
        })}
      </View>

      {/* Custom Actions */}
      <View style={styles.drawerFooter}>
        <DrawerItem
          label="Logout"
          onPress={() => {
            // Handle logout
            navigation.closeDrawer();
          }}
          icon={({ size, color }) => (
            <Icon name="log-out-outline" size={size} color={color} />
          )}
          labelStyle={styles.logoutLabel}
        />
      </View>
    </DrawerContentScrollView>
  );
}

const styles = StyleSheet.create({
  drawerHeader: {
    backgroundColor: '#007bff',
    padding: 20,
    alignItems: 'center',
  },
  userAvatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    marginBottom: 10,
  },
  userName: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  userEmail: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
  },
  drawerSection: {
    marginTop: 15,
  },
  drawerItem: {
    marginVertical: 2,
  },
  drawerItemFocused: {
    backgroundColor: 'rgba(0, 123, 255, 0.1)',
  },
  drawerLabel: {
    fontSize: 16,
    fontWeight: '500',
  },
  drawerLabelFocused: {
    color: '#007bff',
    fontWeight: 'bold',
  },
  drawerFooter: {
    marginTop: 'auto',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    paddingTop: 15,
  },
  logoutLabel: {
    color: '#dc3545',
  },
  badge: {
    backgroundColor: '#dc3545',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
    minWidth: 20,
    alignItems: 'center',
  },
  badgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
});
```

## Nested Navigation

### Complex Navigation Structure
```javascript
// navigation/AppNavigator.js
import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createDrawerNavigator } from '@react-navigation/drawer';

import AuthNavigator from './AuthNavigator';
import MainTabNavigator from './MainTabNavigator';
import ModalScreen from '../screens/ModalScreen';

const Stack = createStackNavigator();
const Drawer = createDrawerNavigator();

// Root Navigator
function AppNavigator() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      {isAuthenticated ? (
        // Main app stack
        <>
          <Stack.Screen name="Main" component={MainDrawerNavigator} />
          <Stack.Screen
            name="Modal"
            component={ModalScreen}
            options={{ presentation: 'modal' }}
          />
        </>
      ) : (
        // Auth stack
        <Stack.Screen name="Auth" component={AuthNavigator} />
      )}
    </Stack.Navigator>
  );
}

// Main Drawer Navigator
function MainDrawerNavigator() {
  return (
    <Drawer.Navigator
      drawerContent={(props) => <CustomDrawerContent {...props} />}
    >
      <Drawer.Screen name="TabNavigator" component={MainTabNavigator} />
      <Drawer.Screen name="Settings" component={SettingsStackNavigator} />
    </Drawer.Navigator>
  );
}

// Main Tab Navigator
function MainTabNavigator() {
  return (
    <Tab.Navigator>
      <Tab.Screen name="HomeStack" component={HomeStackNavigator} />
      <Tab.Screen name="SearchStack" component={SearchStackNavigator} />
      <Tab.Screen name="ProfileStack" component={ProfileStackNavigator} />
    </Tab.Navigator>
  );
}

// Individual Stack Navigators
function HomeStackNavigator() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={HomeScreen} />
      <Stack.Screen name="Details" component={DetailsScreen} />
    </Stack.Navigator>
  );
}

function SearchStackNavigator() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Search" component={SearchScreen} />
      <Stack.Screen name="Results" component={ResultsScreen} />
      <Stack.Screen name="Details" component={DetailsScreen} />
    </Stack.Navigator>
  );
}

function ProfileStackNavigator() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Profile" component={ProfileScreen} />
      <Stack.Screen name="EditProfile" component={EditProfileScreen} />
    </Stack.Navigator>
  );
}
```

## Navigation Patterns

### Navigation Service (Imperative Navigation)
```javascript
// services/NavigationService.js
import { createNavigationContainerRef } from '@react-navigation/native';

export const navigationRef = createNavigationContainerRef();

export function navigate(name, params) {
  if (navigationRef.isReady()) {
    navigationRef.navigate(name, params);
  }
}

export function goBack() {
  if (navigationRef.isReady()) {
    navigationRef.goBack();
  }
}

export function reset(state) {
  if (navigationRef.isReady()) {
    navigationRef.reset(state);
  }
}

export function getCurrentRoute() {
  if (navigationRef.isReady()) {
    return navigationRef.getCurrentRoute();
  }
}

// Usage in App.js
function App() {
  return (
    <NavigationContainer ref={navigationRef}>
      {/* Your navigators */}
    </NavigationContainer>
  );
}

// Usage anywhere in the app
import { navigate } from '../services/NavigationService';

// In a Redux action, utility function, etc.
function someFunction() {
  navigate('Details', { id: 123 });
}
```

### Navigation with Redux
```javascript
// store/navigationSlice.js
import { createSlice } from '@reduxjs/toolkit';

const navigationSlice = createSlice({
  name: 'navigation',
  initialState: {
    currentRoute: null,
    previousRoute: null,
    isNavigating: false,
  },
  reducers: {
    setCurrentRoute: (state, action) => {
      state.previousRoute = state.currentRoute;
      state.currentRoute = action.payload;
    },
    setNavigating: (state, action) => {
      state.isNavigating = action.payload;
    },
  },
});

export const { setCurrentRoute, setNavigating } = navigationSlice.actions;
export default navigationSlice.reducer;

// hooks/useNavigationTracking.js
import { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { useNavigationState } from '@react-navigation/native';
import { setCurrentRoute } from '../store/navigationSlice';

export function useNavigationTracking() {
  const dispatch = useDispatch();
  const navigationState = useNavigationState(state => state);

  useEffect(() => {
    if (navigationState) {
      const currentRoute = getCurrentRouteName(navigationState);
      dispatch(setCurrentRoute(currentRoute));
    }
  }, [navigationState, dispatch]);
}

function getCurrentRouteName(navigationState) {
  if (!navigationState) return null;
  
  const route = navigationState.routes[navigationState.index];
  
  if (route.state) {
    return getCurrentRouteName(route.state);
  }
  
  return route.name;
}
```

## Deep Linking

### Basic Deep Linking Setup
```javascript
// navigation/LinkingConfiguration.js
const linking = {
  prefixes: ['myapp://'], // Custom scheme
  config: {
    screens: {
      Home: 'home',
      Details: 'details/:id',
      Profile: {
        path: 'profile/:userId',
        parse: {
          userId: (userId) => parseInt(userId, 10),
        },
        stringify: {
          userId: (userId) => `${userId}`,
        },
      },
      // Nested navigation
      Main: {
        screens: {
          TabNavigator: {
            screens: {
              HomeStack: {
                screens: {
                  Home: 'home',
                  Details: 'details/:id',
                },
              },
              ProfileStack: {
                screens: {
                  Profile: 'profile',
                  EditProfile: 'profile/edit',
                },
              },
            },
          },
        },
      },
    },
  },
};

// App.js
function App() {
  return (
    <NavigationContainer linking={linking}>
      {/* Your navigators */}
    </NavigationContainer>
  );
}
```

### Advanced Deep Linking with Authentication
```javascript
// hooks/useDeepLinking.js
import { useEffect, useState } from 'react';
import { Linking } from 'react-native';
import { useNavigation } from '@react-navigation/native';

export function useDeepLinking() {
  const navigation = useNavigation();
  const [initialUrl, setInitialUrl] = useState(null);

  useEffect(() => {
    // Handle app launch from deep link
    const getInitialUrl = async () => {
      const url = await Linking.getInitialURL();
      if (url) {
        setInitialUrl(url);
        handleDeepLink(url);
      }
    };

    // Handle deep link when app is already running
    const handleUrlChange = (url) => {
      handleDeepLink(url);
    };

    getInitialUrl();

    const subscription = Linking.addEventListener('url', handleUrlChange);

    return () => {
      subscription?.remove();
    };
  }, []);

  const handleDeepLink = (url) => {
    // Parse URL and extract route information
    const route = parseDeepLink(url);
    
    if (route) {
      // Check if user is authenticated for protected routes
      if (route.requiresAuth && !isAuthenticated()) {
        // Store intended destination and redirect to login
        storeIntendedDestination(route);
        navigation.navigate('Auth', { screen: 'Login' });
      } else {
        // Navigate to the intended route
        navigation.navigate(route.screen, route.params);
      }
    }
  };

  return { initialUrl };
}

function parseDeepLink(url) {
  // Example: myapp://details/123
  const urlParts = url.replace('myapp://', '').split('/');
  
  switch (urlParts[0]) {
    case 'details':
      return {
        screen: 'Details',
        params: { id: urlParts[1] },
        requiresAuth: false,
      };
    case 'profile':
      return {
        screen: 'Profile',
        params: { userId: urlParts[1] },
        requiresAuth: true,
      };
    default:
      return null;
  }
}
```

## Navigation Guards

### Route Protection
```javascript
// components/ProtectedRoute.js
import React from 'react';
import { useSelector } from 'react-redux';
import { useNavigation } from '@react-navigation/native';

function ProtectedRoute({ children, requiredRole = null }) {
  const navigation = useNavigation();
  const { user, isAuthenticated } = useSelector(state => state.auth);

  useEffect(() => {
    if (!isAuthenticated) {
      navigation.replace('Auth', { screen: 'Login' });
      return;
    }

    if (requiredRole && user?.role !== requiredRole) {
      navigation.replace('Unauthorized');
      return;
    }
  }, [isAuthenticated, user, requiredRole, navigation]);

  if (!isAuthenticated) {
    return <LoadingScreen />;
  }

  if (requiredRole && user?.role !== requiredRole) {
    return <UnauthorizedScreen />;
  }

  return children;
}

// Usage in navigator
function ProtectedStackNavigator() {
  return (
    <Stack.Navigator>
      <Stack.Screen 
        name="AdminPanel" 
        options={{ title: 'Admin Panel' }}
      >
        {(props) => (
          <ProtectedRoute requiredRole="admin">
            <AdminPanelScreen {...props} />
          </ProtectedRoute>
        )}
      </Stack.Screen>
    </Stack.Navigator>
  );
}
```

### Navigation Middleware
```javascript
// hooks/useNavigationMiddleware.js
import { useEffect } from 'react';
import { useNavigation, useNavigationState } from '@react-navigation/native';

export function useNavigationMiddleware() {
  const navigation = useNavigation();
  const navigationState = useNavigationState(state => state);

  useEffect(() => {
    const unsubscribe = navigation.addListener('beforeRemove', (e) => {
      // Prevent leaving the screen
      if (shouldPreventNavigation()) {
        e.preventDefault();
        
        // Show confirmation dialog
        Alert.alert(
          'Discard changes?',
          'You have unsaved changes. Are you sure you want to discard them?',
          [
            { text: "Don't leave", style: 'cancel', onPress: () => {} },
            {
              text: 'Discard',
              style: 'destructive',
              onPress: () => navigation.dispatch(e.data.action),
            },
          ]
        );
      }
    });

    return unsubscribe;
  }, [navigation]);

  const shouldPreventNavigation = () => {
    // Your logic to determine if navigation should be prevented
    return hasUnsavedChanges;
  };
}
```

---

*Continue to: [13-react-native-expo.md](./13-react-native-expo.md)*
