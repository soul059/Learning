# React Native Performance Optimization

## Table of Contents
- [Performance Fundamentals](#performance-fundamentals)
- [JavaScript Thread Optimization](#javascript-thread-optimization)
- [UI Thread Optimization](#ui-thread-optimization)
- [Memory Management](#memory-management)
- [Bundle Size Optimization](#bundle-size-optimization)
- [Image Optimization](#image-optimization)
- [List Performance](#list-performance)
- [Animation Performance](#animation-performance)

## Performance Fundamentals

### Understanding React Native Architecture
```javascript
// Performance monitoring setup
import { InteractionManager, Alert } from 'react-native';
import Flipper from 'react-native-flipper';

class PerformanceMonitor {
  static measureExecutionTime(name, func) {
    const start = Date.now();
    const result = func();
    const end = Date.now();
    
    console.log(`${name} took ${end - start}ms`);
    
    // Send to Flipper for debugging
    if (__DEV__) {
      Flipper.sendEvent('performance', {
        name,
        duration: end - start,
        timestamp: Date.now(),
      });
    }
    
    return result;
  }

  static async measureAsyncExecutionTime(name, asyncFunc) {
    const start = Date.now();
    const result = await asyncFunc();
    const end = Date.now();
    
    console.log(`${name} took ${end - start}ms`);
    return result;
  }

  static trackComponentRender(componentName) {
    return function(WrappedComponent) {
      return class extends React.Component {
        componentDidMount() {
          console.log(`${componentName} mounted`);
        }

        componentDidUpdate() {
          console.log(`${componentName} updated`);
        }

        render() {
          const start = performance.now();
          const result = <WrappedComponent {...this.props} />;
          const end = performance.now();
          
          if (end - start > 16) { // Longer than one frame (60fps)
            console.warn(`${componentName} render took ${end - start}ms`);
          }
          
          return result;
        }
      };
    };
  }
}

// Usage
const OptimizedComponent = PerformanceMonitor.trackComponentRender('MyComponent')(MyComponent);
```

### Performance Profiling
```javascript
import React, { Profiler } from 'react';
import { View, Text } from 'react-native';

function onRenderCallback(id, phase, actualDuration, baseDuration, startTime, commitTime) {
  console.log('Profiler Results:', {
    id,
    phase, // "mount" or "update"
    actualDuration, // Time spent rendering the committed update
    baseDuration, // Estimated time to render the entire subtree without memoization
    startTime, // When React began rendering this update
    commitTime, // When React committed this update
  });

  // Log slow renders
  if (actualDuration > 50) {
    console.warn(`Slow render detected in ${id}: ${actualDuration}ms`);
  }

  // Track performance metrics
  if (typeof analytics !== 'undefined') {
    analytics.track('component_render', {
      component: id,
      duration: actualDuration,
      phase,
    });
  }
}

function ProfiledApp() {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <View>
        <Profiler id="Header" onRender={onRenderCallback}>
          <Header />
        </Profiler>
        
        <Profiler id="Content" onRender={onRenderCallback}>
          <Content />
        </Profiler>
        
        <Profiler id="Footer" onRender={onRenderCallback}>
          <Footer />
        </Profiler>
      </View>
    </Profiler>
  );
}

// Performance testing component
function PerformanceTest() {
  const [items, setItems] = useState([]);
  const [renderCount, setRenderCount] = useState(0);

  const addItems = (count) => {
    const newItems = Array.from({ length: count }, (_, i) => ({
      id: Date.now() + i,
      name: `Item ${Date.now() + i}`,
      value: Math.random(),
    }));
    
    PerformanceMonitor.measureExecutionTime('Add Items', () => {
      setItems(prev => [...prev, ...newItems]);
    });
  };

  const clearItems = () => {
    PerformanceMonitor.measureExecutionTime('Clear Items', () => {
      setItems([]);
    });
  };

  useEffect(() => {
    setRenderCount(prev => prev + 1);
  });

  return (
    <View style={styles.container}>
      <Text>Render Count: {renderCount}</Text>
      <Text>Items: {items.length}</Text>
      
      <Button title="Add 100 Items" onPress={() => addItems(100)} />
      <Button title="Add 1000 Items" onPress={() => addItems(1000)} />
      <Button title="Clear Items" onPress={clearItems} />
    </View>
  );
}
```

## JavaScript Thread Optimization

### Optimizing Heavy Computations
```javascript
import React, { useState, useMemo, useCallback } from 'react';
import { View, Text, Button, InteractionManager } from 'react-native';

// Heavy computation example
function heavyComputation(data) {
  console.log('Starting heavy computation...');
  let result = 0;
  
  for (let i = 0; i < data.length; i++) {
    for (let j = 0; j < 1000; j++) {
      result += Math.sqrt(data[i] * j);
    }
  }
  
  return result;
}

// Optimized with useMemo
function OptimizedHeavyComponent({ data }) {
  const expensiveValue = useMemo(() => {
    return heavyComputation(data);
  }, [data]);

  return (
    <View>
      <Text>Computed Result: {expensiveValue}</Text>
    </View>
  );
}

// Breaking work into chunks
function ChunkedComputation({ data, onProgress, onComplete }) {
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);

  const processChunk = useCallback((startIndex, chunkSize, accumulator = 0) => {
    const endIndex = Math.min(startIndex + chunkSize, data.length);
    
    // Process chunk
    for (let i = startIndex; i < endIndex; i++) {
      accumulator += Math.sqrt(data[i]);
    }

    const newProgress = endIndex / data.length;
    setProgress(newProgress);
    onProgress?.(newProgress);

    if (endIndex < data.length) {
      // Continue with next chunk after allowing UI to update
      InteractionManager.runAfterInteractions(() => {
        requestAnimationFrame(() => {
          processChunk(endIndex, chunkSize, accumulator);
        });
      });
    } else {
      // Computation complete
      setResult(accumulator);
      onComplete?.(accumulator);
    }
  }, [data, onProgress, onComplete]);

  const startChunkedComputation = useCallback(() => {
    setProgress(0);
    setResult(null);
    processChunk(0, 100); // Process 100 items per chunk
  }, [processChunk]);

  return (
    <View style={styles.container}>
      <Text>Progress: {Math.round(progress * 100)}%</Text>
      {result && <Text>Result: {result}</Text>}
      <Button title="Start Computation" onPress={startChunkedComputation} />
    </View>
  );
}

// Web Worker alternative using separate thread
class WorkerManager {
  static async runInWorker(workerScript, data) {
    return new Promise((resolve, reject) => {
      if (Platform.OS === 'web') {
        // Use Web Worker on web platform
        const worker = new Worker(workerScript);
        worker.postMessage(data);
        
        worker.onmessage = (e) => {
          resolve(e.data);
          worker.terminate();
        };
        
        worker.onerror = (error) => {
          reject(error);
          worker.terminate();
        };
      } else {
        // Fallback to chunked processing on mobile
        this.processInChunks(data).then(resolve).catch(reject);
      }
    });
  }

  static async processInChunks(data, chunkSize = 1000) {
    let result = 0;
    
    for (let i = 0; i < data.length; i += chunkSize) {
      const chunk = data.slice(i, i + chunkSize);
      
      // Process chunk
      for (const item of chunk) {
        result += Math.sqrt(item);
      }
      
      // Yield control back to UI thread
      await new Promise(resolve => {
        InteractionManager.runAfterInteractions(resolve);
      });
    }
    
    return result;
  }
}
```

### Debouncing and Throttling
```javascript
import { useCallback, useRef, useEffect } from 'react';

// Debounce hook
function useDebounce(callback, delay) {
  const timeoutRef = useRef(null);

  const debouncedCallback = useCallback((...args) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    timeoutRef.current = setTimeout(() => {
      callback(...args);
    }, delay);
  }, [callback, delay]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return debouncedCallback;
}

// Throttle hook
function useThrottle(callback, delay) {
  const lastRun = useRef(Date.now());

  const throttledCallback = useCallback((...args) => {
    if (Date.now() - lastRun.current >= delay) {
      callback(...args);
      lastRun.current = Date.now();
    }
  }, [callback, delay]);

  return throttledCallback;
}

// Usage example
function SearchComponent() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  const search = async (searchQuery) => {
    if (!searchQuery) {
      setResults([]);
      return;
    }

    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error('Search error:', error);
    }
  };

  const debouncedSearch = useDebounce(search, 300);

  const handleQueryChange = (text) => {
    setQuery(text);
    debouncedSearch(text);
  };

  return (
    <View>
      <TextInput
        value={query}
        onChangeText={handleQueryChange}
        placeholder="Search..."
      />
      <FlatList
        data={results}
        renderItem={({ item }) => <SearchResult item={item} />}
        keyExtractor={item => item.id}
      />
    </View>
  );
}
```

## UI Thread Optimization

### Optimizing Animations
```javascript
import React, { useRef } from 'react';
import { Animated, PanGestureHandler, State } from 'react-native-reanimated';

// Use native driver for better performance
function OptimizedAnimation() {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const fadeIn = () => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 300,
      useNativeDriver: true, // Essential for performance
    }).start();
  };

  const pulseAnimation = () => {
    Animated.sequence([
      Animated.timing(scaleAnim, {
        toValue: 1.2,
        duration: 150,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 150,
        useNativeDriver: true,
      }),
    ]).start();
  };

  return (
    <Animated.View
      style={{
        opacity: fadeAnim,
        transform: [{ scale: scaleAnim }],
      }}
    >
      <Text>Optimized Animation</Text>
    </Animated.View>
  );
}

// Reanimated 2 for complex animations
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  useAnimatedGestureHandler,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';

function ReanimatedOptimized() {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const scale = useSharedValue(1);

  const gestureHandler = useAnimatedGestureHandler({
    onStart: (_, context) => {
      context.startX = translateX.value;
      context.startY = translateY.value;
      scale.value = withSpring(1.1);
    },
    onActive: (event, context) => {
      translateX.value = context.startX + event.translationX;
      translateY.value = context.startY + event.translationY;
    },
    onEnd: () => {
      translateX.value = withSpring(0);
      translateY.value = withSpring(0);
      scale.value = withSpring(1);
    },
  });

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
        { scale: scale.value },
      ],
    };
  });

  return (
    <PanGestureHandler onGestureEvent={gestureHandler}>
      <Animated.View style={[styles.box, animatedStyle]}>
        <Text>Drag me!</Text>
      </Animated.View>
    </PanGestureHandler>
  );
}
```

### Reducing Bridge Communication
```javascript
// Bad: Frequent bridge communication
function BadScrollHandler() {
  const [scrollY, setScrollY] = useState(0);

  const handleScroll = (event) => {
    setScrollY(event.nativeEvent.contentOffset.y); // Bridge call on every frame
  };

  return (
    <ScrollView onScroll={handleScroll} scrollEventThrottle={16}>
      <Text>Scroll Y: {scrollY}</Text>
      {/* Content */}
    </ScrollView>
  );
}

// Good: Use Animated.event to reduce bridge calls
function GoodScrollHandler() {
  const scrollY = useRef(new Animated.Value(0)).current;

  const animatedStyle = {
    transform: [
      {
        translateY: scrollY.interpolate({
          inputRange: [0, 100],
          outputRange: [0, -50],
          extrapolate: 'clamp',
        }),
      },
    ],
  };

  return (
    <View>
      <Animated.View style={[styles.header, animatedStyle]}>
        <Text>Animated Header</Text>
      </Animated.View>
      
      <ScrollView
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: true }
        )}
        scrollEventThrottle={16}
      >
        {/* Content */}
      </ScrollView>
    </View>
  );
}

// Minimize state updates in render
function OptimizedComponent({ data }) {
  // Expensive computation moved to useMemo
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      computed: heavyComputation(item),
    }));
  }, [data]);

  // Stable callback reference
  const handlePress = useCallback((item) => {
    // Handle press without recreating function
    console.log('Pressed:', item.id);
  }, []);

  return (
    <FlatList
      data={processedData}
      renderItem={({ item }) => (
        <OptimizedListItem item={item} onPress={handlePress} />
      )}
      keyExtractor={item => item.id}
      getItemLayout={(data, index) => ({
        length: ITEM_HEIGHT,
        offset: ITEM_HEIGHT * index,
        index,
      })}
    />
  );
}
```

## Memory Management

### Preventing Memory Leaks
```javascript
import React, { useEffect, useRef, useState } from 'react';

function MemoryOptimizedComponent() {
  const [data, setData] = useState([]);
  const isMountedRef = useRef(true);
  const timeoutRef = useRef(null);
  const subscriptionRef = useRef(null);

  useEffect(() => {
    // Cleanup function to prevent memory leaks
    return () => {
      isMountedRef.current = false;
      
      // Clear timeouts
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      // Unsubscribe from listeners
      if (subscriptionRef.current) {
        subscriptionRef.current.remove();
      }
    };
  }, []);

  const fetchData = async () => {
    try {
      const response = await fetch('/api/data');
      const result = await response.json();
      
      // Only update state if component is still mounted
      if (isMountedRef.current) {
        setData(result);
      }
    } catch (error) {
      if (isMountedRef.current) {
        console.error('Fetch error:', error);
      }
    }
  };

  const scheduleUpdate = () => {
    timeoutRef.current = setTimeout(() => {
      if (isMountedRef.current) {
        fetchData();
      }
    }, 5000);
  };

  // Image cache management
  const [imageCache, setImageCache] = useState(new Map());
  const maxCacheSize = 50;

  const addToImageCache = (uri, image) => {
    setImageCache(prev => {
      const newCache = new Map(prev);
      
      // Remove oldest entries if cache is full
      if (newCache.size >= maxCacheSize) {
        const firstKey = newCache.keys().next().value;
        newCache.delete(firstKey);
      }
      
      newCache.set(uri, image);
      return newCache;
    });
  };

  const clearImageCache = () => {
    setImageCache(new Map());
  };

  return (
    <View>
      {/* Component content */}
    </View>
  );
}

// Memory-efficient image component
function OptimizedImage({ uri, style, ...props }) {
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(true);
  const isMountedRef = useRef(true);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    loadImage();
  }, [uri]);

  const loadImage = async () => {
    if (!uri) return;

    try {
      setLoading(true);
      
      // Check cache first
      const cached = ImageCache.get(uri);
      if (cached) {
        if (isMountedRef.current) {
          setImageData(cached);
          setLoading(false);
        }
        return;
      }

      // Load and cache image
      const image = await ImageCache.load(uri);
      
      if (isMountedRef.current) {
        setImageData(image);
        setLoading(false);
      }
    } catch (error) {
      if (isMountedRef.current) {
        setLoading(false);
        console.error('Image load error:', error);
      }
    }
  };

  if (loading) {
    return <View style={[style, { backgroundColor: '#f0f0f0' }]} />;
  }

  return (
    <Image
      source={{ uri: imageData }}
      style={style}
      {...props}
    />
  );
}

// Memory monitoring
class MemoryMonitor {
  static logMemoryUsage() {
    if (global.performance && global.performance.memory) {
      const memory = global.performance.memory;
      console.log('Memory Usage:', {
        used: Math.round(memory.usedJSHeapSize / 1048576),
        total: Math.round(memory.totalJSHeapSize / 1048576),
        limit: Math.round(memory.jsHeapSizeLimit / 1048576),
      });
    }
  }

  static startMemoryMonitoring(interval = 10000) {
    return setInterval(() => {
      this.logMemoryUsage();
    }, interval);
  }
}
```

## Bundle Size Optimization

### Code Splitting and Lazy Loading
```javascript
import React, { Suspense, lazy } from 'react';
import { ActivityIndicator } from 'react-native';

// Lazy load screens
const HomeScreen = lazy(() => import('./screens/HomeScreen'));
const ProfileScreen = lazy(() => import('./screens/ProfileScreen'));
const SettingsScreen = lazy(() => import('./screens/SettingsScreen'));

// Loading component
function LoadingScreen() {
  return (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color="#007bff" />
      <Text>Loading...</Text>
    </View>
  );
}

// App with lazy loading
function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home">
          {() => (
            <Suspense fallback={<LoadingScreen />}>
              <HomeScreen />
            </Suspense>
          )}
        </Stack.Screen>
        
        <Stack.Screen name="Profile">
          {() => (
            <Suspense fallback={<LoadingScreen />}>
              <ProfileScreen />
            </Suspense>
          )}
        </Stack.Screen>
        
        <Stack.Screen name="Settings">
          {() => (
            <Suspense fallback={<LoadingScreen />}>
              <SettingsScreen />
            </Suspense>
          )}
        </Stack.Screen>
      </Stack.Navigator>
    </NavigationContainer>
  );
}

// Dynamic imports for large libraries
async function loadLargeLibrary() {
  const { default: LargeLibrary } = await import('large-library');
  return LargeLibrary;
}

// Bundle analyzer setup (metro.config.js)
const { getDefaultConfig } = require('@expo/metro-config');

const config = getDefaultConfig(__dirname);

// Enable bundle splitting
config.serializer = {
  ...config.serializer,
  createModuleIdFactory: () => (path) => {
    // Create consistent module IDs for better caching
    return require('crypto')
      .createHash('sha1')
      .update(path)
      .digest('hex')
      .substr(0, 8);
  },
};

// Tree shaking optimization
// Import only what you need
import { debounce } from 'lodash/debounce'; // ✅ Good
// import _ from 'lodash'; // ❌ Bad - imports entire library
```

### Image and Asset Optimization
```javascript
// Image optimization
const ImageOptimizer = {
  // Resize images based on screen density
  getOptimizedImageUri(baseUri, width, height) {
    const { scale } = Dimensions.get('window');
    const optimizedWidth = width * scale;
    const optimizedHeight = height * scale;
    
    return `${baseUri}?w=${optimizedWidth}&h=${optimizedHeight}&q=80`;
  },

  // Progressive image loading
  async loadProgressiveImage(uri) {
    // Load low quality first
    const lowQualityUri = `${uri}?w=50&q=20`;
    const highQualityUri = `${uri}?q=80`;
    
    return {
      placeholder: lowQualityUri,
      source: highQualityUri,
    };
  },

  // WebP support detection
  supportsWebP() {
    return Platform.OS === 'android' || 
           (Platform.OS === 'ios' && parseInt(Platform.Version, 10) >= 14);
  },

  getImageFormat(uri) {
    if (this.supportsWebP()) {
      return uri.replace(/\.(jpg|jpeg|png)$/, '.webp');
    }
    return uri;
  },
};

// Optimized image component
function OptimizedImageComponent({ uri, width, height, style, ...props }) {
  const [imageSource, setImageSource] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadOptimizedImage();
  }, [uri, width, height]);

  const loadOptimizedImage = async () => {
    try {
      const optimizedUri = ImageOptimizer.getOptimizedImageUri(uri, width, height);
      const webpUri = ImageOptimizer.getImageFormat(optimizedUri);
      
      const { placeholder, source } = await ImageOptimizer.loadProgressiveImage(webpUri);
      
      // Load placeholder first
      setImageSource(placeholder);
      setIsLoading(false);
      
      // Then load high quality
      setTimeout(() => {
        setImageSource(source);
      }, 100);
      
    } catch (error) {
      console.error('Image optimization error:', error);
      setImageSource(uri);
      setIsLoading(false);
    }
  };

  return (
    <Image
      source={{ uri: imageSource }}
      style={[{ width, height }, style]}
      {...props}
    />
  );
}
```

## List Performance

### FlatList Optimization
```javascript
import React, { useMemo, useCallback } from 'react';
import { FlatList, View, Text, StyleSheet } from 'react-native';

// Optimized list item
const OptimizedListItem = React.memo(({ item, onPress }) => {
  const handlePress = useCallback(() => {
    onPress(item);
  }, [item, onPress]);

  return (
    <TouchableOpacity style={styles.item} onPress={handlePress}>
      <Text style={styles.title}>{item.title}</Text>
      <Text style={styles.subtitle}>{item.subtitle}</Text>
    </TouchableOpacity>
  );
});

// Optimized FlatList
function OptimizedFlatList({ data, onItemPress }) {
  const keyExtractor = useCallback((item) => item.id.toString(), []);
  
  const renderItem = useCallback(({ item }) => (
    <OptimizedListItem item={item} onPress={onItemPress} />
  ), [onItemPress]);

  const getItemLayout = useCallback((data, index) => ({
    length: ITEM_HEIGHT,
    offset: ITEM_HEIGHT * index,
    index,
  }), []);

  // Memoize data processing
  const processedData = useMemo(() => {
    return data.filter(item => item.visible !== false);
  }, [data]);

  return (
    <FlatList
      data={processedData}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      getItemLayout={getItemLayout}
      
      // Performance optimizations
      removeClippedSubviews={true}
      maxToRenderPerBatch={10}
      updateCellsBatchingPeriod={50}
      initialNumToRender={20}
      windowSize={10}
      
      // Memory optimizations
      onEndReachedThreshold={0.5}
      onEndReached={loadMore}
      
      // Separator component
      ItemSeparatorComponent={() => <View style={styles.separator} />}
      
      // Empty state
      ListEmptyComponent={() => (
        <View style={styles.emptyState}>
          <Text>No items found</Text>
        </View>
      )}
      
      // Header and footer
      ListHeaderComponent={() => (
        <View style={styles.header}>
          <Text>List Header</Text>
        </View>
      )}
      
      ListFooterComponent={() => (
        <View style={styles.footer}>
          <Text>List Footer</Text>
        </View>
      )}
    />
  );
}

// Virtual list for large datasets
import { VirtualizedList } from 'react-native';

function VirtualizedListComponent({ data }) {
  const getItem = (data, index) => data[index];
  const getItemCount = (data) => data.length;

  const renderItem = ({ item, index }) => (
    <View style={[styles.item, { height: getItemHeight(index) }]}>
      <Text>{item.title}</Text>
    </View>
  );

  const getItemHeight = (index) => {
    // Variable height logic
    return index % 2 === 0 ? 80 : 120;
  };

  return (
    <VirtualizedList
      data={data}
      initialNumToRender={10}
      renderItem={renderItem}
      keyExtractor={(item, index) => item.id}
      getItemCount={getItemCount}
      getItem={getItem}
      getItemLayout={(data, index) => ({
        length: getItemHeight(index),
        offset: calculateOffset(index),
        index,
      })}
    />
  );
}

// Section list optimization
import { SectionList } from 'react-native';

function OptimizedSectionList({ sections }) {
  const renderSectionHeader = useCallback(({ section }) => (
    <View style={styles.sectionHeader}>
      <Text style={styles.sectionTitle}>{section.title}</Text>
    </View>
  ), []);

  const renderItem = useCallback(({ item, section }) => (
    <OptimizedListItem item={item} onPress={handleItemPress} />
  ), []);

  const keyExtractor = useCallback((item, index) => `${item.id}-${index}`, []);

  return (
    <SectionList
      sections={sections}
      renderItem={renderItem}
      renderSectionHeader={renderSectionHeader}
      keyExtractor={keyExtractor}
      
      // Performance optimizations
      removeClippedSubviews={true}
      maxToRenderPerBatch={5}
      updateCellsBatchingPeriod={100}
      initialNumToRender={10}
      windowSize={5}
      
      // Sticky headers
      stickySectionHeadersEnabled={true}
      
      // Section separator
      SectionSeparatorComponent={() => <View style={styles.sectionSeparator} />}
    />
  );
}
```

## Animation Performance

### High-Performance Animations
```javascript
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withSequence,
  withRepeat,
  runOnJS,
  interpolate,
  Extrapolate,
} from 'react-native-reanimated';

function HighPerformanceAnimations() {
  const progress = useSharedValue(0);
  const scale = useSharedValue(1);
  const rotation = useSharedValue(0);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        {
          scale: interpolate(
            progress.value,
            [0, 0.5, 1],
            [1, 1.2, 1],
            Extrapolate.CLAMP
          ),
        },
        {
          rotate: `${interpolate(
            progress.value,
            [0, 1],
            [0, 360],
            Extrapolate.CLAMP
          )}deg`,
        },
      ],
      opacity: interpolate(
        progress.value,
        [0, 0.5, 1],
        [1, 0.8, 1],
        Extrapolate.CLAMP
      ),
    };
  }, []);

  const startAnimation = () => {
    progress.value = withSequence(
      withTiming(0.5, { duration: 500 }),
      withSpring(1, { damping: 10, stiffness: 100 })
    );
  };

  const startInfiniteRotation = () => {
    rotation.value = withRepeat(
      withTiming(360, { duration: 2000 }),
      -1, // Infinite
      false // Don't reverse
    );
  };

  return (
    <View style={styles.container}>
      <Animated.View style={[styles.animatedBox, animatedStyle]}>
        <Text>Animated Element</Text>
      </Animated.View>
      
      <Button title="Start Animation" onPress={startAnimation} />
      <Button title="Start Rotation" onPress={startInfiniteRotation} />
    </View>
  );
}

// Gesture-driven animations
import { PanGestureHandler, State } from 'react-native-gesture-handler';

function GestureDrivenAnimation() {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const opacity = useSharedValue(1);

  const gestureHandler = useAnimatedGestureHandler({
    onStart: (_, context) => {
      context.startX = translateX.value;
      context.startY = translateY.value;
    },
    onActive: (event, context) => {
      translateX.value = context.startX + event.translationX;
      translateY.value = context.startY + event.translationY;
      
      // Update opacity based on distance from center
      const distance = Math.sqrt(
        Math.pow(translateX.value, 2) + Math.pow(translateY.value, 2)
      );
      opacity.value = interpolate(
        distance,
        [0, 200],
        [1, 0.3],
        Extrapolate.CLAMP
      );
    },
    onEnd: (event) => {
      const shouldDismiss = Math.abs(event.velocityX) > 500 || 
                           Math.abs(event.velocityY) > 500;
      
      if (shouldDismiss) {
        translateX.value = withTiming(event.velocityX > 0 ? 1000 : -1000);
        translateY.value = withTiming(event.velocityY > 0 ? 1000 : -1000);
        opacity.value = withTiming(0, {}, () => {
          runOnJS(onDismiss)();
        });
      } else {
        translateX.value = withSpring(0);
        translateY.value = withSpring(0);
        opacity.value = withSpring(1);
      }
    },
  });

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
      ],
      opacity: opacity.value,
    };
  });

  const onDismiss = () => {
    console.log('Card dismissed');
    // Reset position
    translateX.value = 0;
    translateY.value = 0;
    opacity.value = 1;
  };

  return (
    <PanGestureHandler onGestureEvent={gestureHandler}>
      <Animated.View style={[styles.card, animatedStyle]}>
        <Text>Drag me to dismiss</Text>
      </Animated.View>
    </PanGestureHandler>
  );
}

// Layout animations
import { Layout, FadeIn, FadeOut, SlideInUp, SlideOutDown } from 'react-native-reanimated';

function LayoutAnimations() {
  const [items, setItems] = useState([1, 2, 3]);

  const addItem = () => {
    setItems(prev => [...prev, Date.now()]);
  };

  const removeItem = (id) => {
    setItems(prev => prev.filter(item => item !== id));
  };

  return (
    <View style={styles.container}>
      <Button title="Add Item" onPress={addItem} />
      
      {items.map(item => (
        <Animated.View
          key={item}
          entering={SlideInUp}
          exiting={SlideOutDown}
          layout={Layout.springify()}
          style={styles.listItem}
        >
          <Text>Item {item}</Text>
          <Button title="Remove" onPress={() => removeItem(item)} />
        </Animated.View>
      ))}
    </View>
  );
}
```

---

*Continue to: [16-react-native-deployment.md](./16-react-native-deployment.md)*
