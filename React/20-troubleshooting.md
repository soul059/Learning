# Common Issues and Troubleshooting Guide

## Table of Contents
- [React Common Issues](#react-common-issues)
- [React Native Troubleshooting](#react-native-troubleshooting)
- [Build and Development Issues](#build-and-development-issues)
- [Performance Problems](#performance-problems)
- [Debugging Strategies](#debugging-strategies)
- [Error Handling Patterns](#error-handling-patterns)
- [Memory Leaks and Cleanup](#memory-leaks-and-cleanup)
- [Platform-Specific Issues](#platform-specific-issues)

## React Common Issues

### State Management Problems
```typescript
// ❌ Problem: State updates not reflecting immediately
const [count, setCount] = useState(0);

const handleClick = () => {
  setCount(count + 1);
  console.log(count); // Still shows old value!
};

// ✅ Solution: Use functional updates or useEffect
const handleClick = () => {
  setCount(prevCount => {
    const newCount = prevCount + 1;
    console.log('New count:', newCount);
    return newCount;
  });
};

// Or use useEffect to respond to state changes
useEffect(() => {
  console.log('Count updated:', count);
}, [count]);

// ❌ Problem: Mutating state directly
const [items, setItems] = useState([]);

const addItem = (item) => {
  items.push(item); // Mutating state!
  setItems(items);
};

// ✅ Solution: Create new state objects
const addItem = (item) => {
  setItems(prevItems => [...prevItems, item]);
};

const updateItem = (index, newItem) => {
  setItems(prevItems => 
    prevItems.map((item, i) => i === index ? newItem : item)
  );
};

const removeItem = (index) => {
  setItems(prevItems => prevItems.filter((_, i) => i !== index));
};

// ❌ Problem: Stale closures in useEffect
const [count, setCount] = useState(0);

useEffect(() => {
  const interval = setInterval(() => {
    setCount(count + 1); // Always uses initial count value!
  }, 1000);
  
  return () => clearInterval(interval);
}, []); // Empty dependency array causes stale closure

// ✅ Solution: Use functional updates or include dependencies
useEffect(() => {
  const interval = setInterval(() => {
    setCount(prevCount => prevCount + 1); // Uses current value
  }, 1000);
  
  return () => clearInterval(interval);
}, []);

// Or include count in dependencies (but this recreates interval)
useEffect(() => {
  const interval = setInterval(() => {
    setCount(count + 1);
  }, 1000);
  
  return () => clearInterval(interval);
}, [count]);
```

### Effect and Dependency Issues
```typescript
// ❌ Problem: Missing dependencies in useEffect
const [user, setUser] = useState(null);
const [posts, setPosts] = useState([]);

useEffect(() => {
  if (user) {
    fetchUserPosts(user.id).then(setPosts); // user.id is a dependency!
  }
}, []); // Missing user dependency

// ✅ Solution: Include all dependencies
useEffect(() => {
  if (user) {
    fetchUserPosts(user.id).then(setPosts);
  }
}, [user]); // Include user dependency

// ❌ Problem: Object/array dependencies causing infinite loops
const [filters, setFilters] = useState({ category: 'all', sort: 'date' });

useEffect(() => {
  fetchData(filters).then(setData);
}, [filters]); // filters object recreated on every render

// ✅ Solution: Use useMemo for object dependencies
const filters = useMemo(() => ({
  category: 'all',
  sort: 'date'
}), []);

// Or use individual primitive dependencies
const [category, setCategory] = useState('all');
const [sort, setSort] = useState('date');

useEffect(() => {
  fetchData({ category, sort }).then(setData);
}, [category, sort]);

// ❌ Problem: Effect cleanup not working
useEffect(() => {
  const subscription = subscribe(callback);
  // Missing cleanup!
}, []);

// ✅ Solution: Always return cleanup function
useEffect(() => {
  const subscription = subscribe(callback);
  
  return () => {
    subscription.unsubscribe();
  };
}, [callback]);

// Advanced cleanup pattern
useEffect(() => {
  let isMounted = true;
  
  const fetchData = async () => {
    try {
      const data = await api.getData();
      if (isMounted) {
        setData(data);
      }
    } catch (error) {
      if (isMounted) {
        setError(error);
      }
    }
  };
  
  fetchData();
  
  return () => {
    isMounted = false;
  };
}, []);
```

### Component Re-rendering Issues
```typescript
// ❌ Problem: Unnecessary re-renders due to inline objects/functions
const MyComponent = ({ data }) => {
  return (
    <ExpensiveChild 
      config={{ theme: 'dark', debug: true }} // New object every render!
      onUpdate={() => console.log('updated')} // New function every render!
      data={data}
    />
  );
};

// ✅ Solution: Use useMemo and useCallback
const MyComponent = ({ data }) => {
  const config = useMemo(() => ({
    theme: 'dark',
    debug: true
  }), []);

  const handleUpdate = useCallback(() => {
    console.log('updated');
  }, []);

  return (
    <ExpensiveChild 
      config={config}
      onUpdate={handleUpdate}
      data={data}
    />
  );
};

// ✅ Alternative: Move static objects outside component
const DEFAULT_CONFIG = { theme: 'dark', debug: true };

const MyComponent = ({ data }) => {
  const handleUpdate = useCallback(() => {
    console.log('updated');
  }, []);

  return (
    <ExpensiveChild 
      config={DEFAULT_CONFIG}
      onUpdate={handleUpdate}
      data={data}
    />
  );
};

// ❌ Problem: React.memo not working due to complex props
const ExpensiveChild = React.memo(({ items, onItemClick }) => {
  // Component logic
});

const Parent = () => {
  const [filter, setFilter] = useState('');
  
  const items = data.filter(item => item.name.includes(filter)); // New array every render!
  
  return (
    <ExpensiveChild 
      items={items}
      onItemClick={(id) => console.log(id)} // New function every render!
    />
  );
};

// ✅ Solution: Memoize props properly
const Parent = () => {
  const [filter, setFilter] = useState('');
  
  const filteredItems = useMemo(() => 
    data.filter(item => item.name.includes(filter)), 
    [filter]
  );
  
  const handleItemClick = useCallback((id) => {
    console.log(id);
  }, []);
  
  return (
    <ExpensiveChild 
      items={filteredItems}
      onItemClick={handleItemClick}
    />
  );
};
```

## React Native Troubleshooting

### Metro and Build Issues
```bash
# ❌ Problem: Metro cache issues
# Symptoms: Changes not reflecting, build errors after dependency updates

# ✅ Solutions: Clear Metro cache
npx react-native start --reset-cache
# or
npm start -- --reset-cache

# Clear all caches
cd android && ./gradlew clean && cd ..
cd ios && xcodebuild clean && cd ..
rm -rf node_modules
npm install

# Clear React Native cache
npx react-native-clean-project
```

### Android Build Problems
```bash
# ❌ Problem: Android build failures

# ✅ Solution 1: Gradle wrapper issues
cd android
./gradlew clean
./gradlew assembleDebug

# ✅ Solution 2: SDK/NDK version mismatches
# Check android/local.properties
sdk.dir=/Users/username/Library/Android/sdk
ndk.dir=/Users/username/Library/Android/sdk/ndk/21.4.7075529

# ✅ Solution 3: Java version issues
# Check gradle.properties
org.gradle.jvmargs=-Xmx2048m -XX:MaxPermSize=512m -XX:+HeapDumpOnOutOfMemoryError -Dfile.encoding=UTF-8

# ✅ Solution 4: Multidex issues
# In android/app/build.gradle
android {
    defaultConfig {
        multiDexEnabled true
    }
}

dependencies {
    implementation 'androidx.multidex:multidex:2.0.1'
}
```

### iOS Build Problems
```bash
# ❌ Problem: iOS build failures

# ✅ Solution 1: CocoaPods issues
cd ios
pod deintegrate
pod install
# or
pod install --repo-update

# ✅ Solution 2: Xcode project corruption
cd ios
rm -rf build/
rm -rf ~/Library/Developer/Xcode/DerivedData/
xcodebuild clean

# ✅ Solution 3: Code signing issues
# In Xcode: Project Settings > Signing & Capabilities
# Select your team and ensure provisioning profile is valid

# ✅ Solution 4: Simulator issues
xcrun simctl erase all
npx react-native run-ios --simulator="iPhone 14"
```

### JavaScript Thread Blocking
```typescript
// ❌ Problem: Blocking JavaScript thread with heavy computations
const processLargeDataset = (data) => {
  // This blocks the JS thread!
  return data.map(item => {
    // Heavy computation
    for (let i = 0; i < 10000; i++) {
      // Some expensive operation
    }
    return transformItem(item);
  });
};

// ✅ Solution 1: Use InteractionManager
import { InteractionManager } from 'react-native';

const processLargeDataset = async (data) => {
  await InteractionManager.runAfterInteractions();
  
  return new Promise((resolve) => {
    // Process in chunks
    const processChunk = (startIndex) => {
      const chunkSize = 100;
      const endIndex = Math.min(startIndex + chunkSize, data.length);
      
      for (let i = startIndex; i < endIndex; i++) {
        // Process item
      }
      
      if (endIndex < data.length) {
        setTimeout(() => processChunk(endIndex), 0);
      } else {
        resolve(processedData);
      }
    };
    
    processChunk(0);
  });
};

// ✅ Solution 2: Use Web Workers (with react-native-workers)
// worker.js
const processData = (data) => {
  // Heavy computation in worker thread
  return data.map(transformItem);
};

self.onmessage = (e) => {
  const result = processData(e.data);
  self.postMessage(result);
};

// main.js
import Worker from './worker.js';

const processWithWorker = (data) => {
  return new Promise((resolve, reject) => {
    const worker = new Worker();
    
    worker.onmessage = (e) => {
      resolve(e.data);
      worker.terminate();
    };
    
    worker.onerror = (error) => {
      reject(error);
      worker.terminate();
    };
    
    worker.postMessage(data);
  });
};

// ✅ Solution 3: Use native modules for heavy computations
// See previous sections on native module development
```

### Navigation Issues
```typescript
// ❌ Problem: Navigation state not updating
const navigate = useNavigation();

const handlePress = () => {
  navigate.navigate('Details');
  // Trying to access route immediately
  console.log(navigate.getState()); // May show old state
};

// ✅ Solution: Use navigation listeners
const navigation = useNavigation();

useEffect(() => {
  const unsubscribe = navigation.addListener('state', (e) => {
    console.log('Navigation state changed:', e.data.state);
  });

  return unsubscribe;
}, [navigation]);

// ❌ Problem: Passing complex objects through navigation
navigate.navigate('Details', { 
  user: complexUserObject // This gets serialized!
});

// ✅ Solution: Pass IDs and fetch data in destination
navigate.navigate('Details', { userId: user.id });

// In Details screen:
const { userId } = route.params;
const user = useSelector(state => selectUserById(state, userId));

// ❌ Problem: Deep linking not working
// Missing URL schemes or intent filters

// ✅ Solution: Proper deep link configuration
// android/app/src/main/AndroidManifest.xml
<intent-filter android:autoVerify="true">
    <action android:name="android.intent.action.VIEW" />
    <category android:name="android.intent.category.DEFAULT" />
    <category android:name="android.intent.category.BROWSABLE" />
    <data android:scheme="myapp" android:host="details" />
</intent-filter>

// ios/MyApp/Info.plist
<key>CFBundleURLTypes</key>
<array>
    <dict>
        <key>CFBundleURLName</key>
        <string>myapp</string>
        <key>CFBundleURLSchemes</key>
        <array>
            <string>myapp</string>
        </array>
    </dict>
</array>
```

## Build and Development Issues

### Vite Configuration Problems
```typescript
// ❌ Problem: Import path resolution issues
import { Button } from '../../../components/Button'; // Relative path hell

// ✅ Solution: Configure path aliases in vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@assets': path.resolve(__dirname, './src/assets'),
    },
  },
  // Fix for some dependencies
  define: {
    global: 'globalThis',
  },
  // Optimize dependencies
  optimizeDeps: {
    include: ['react', 'react-dom'],
    exclude: ['@vite/client', '@vite/env'],
  },
});

// Now you can use:
import { Button } from '@components/Button';

// ❌ Problem: Environment variables not working
console.log(process.env.REACT_APP_API_URL); // undefined in Vite

// ✅ Solution: Use Vite env variables
console.log(import.meta.env.VITE_API_URL);

// .env file
VITE_API_URL=http://localhost:3000
VITE_APP_TITLE=My App

// vite-env.d.ts
interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_TITLE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

### TypeScript Configuration Issues
```json
// ❌ Problem: TypeScript errors in build but not in IDE
// tsconfig.json with loose settings
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": false
  }
}

// ✅ Solution: Strict TypeScript configuration
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["DOM", "DOM.Iterable", "ES6"],
    "allowJs": true,
    "skipLibCheck": false,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"]
    }
  },
  "include": [
    "src/**/*",
    "src/**/*.json"
  ],
  "exclude": [
    "node_modules",
    "build",
    "dist"
  ]
}

// ❌ Problem: Module not found errors
import { myFunction } from './utils'; // Missing .ts extension in some configs

// ✅ Solution: Explicit extensions or proper module resolution
import { myFunction } from './utils.ts';
// or configure module resolution properly

// ❌ Problem: Conflicting type definitions
// Multiple @types packages or global type conflicts

// ✅ Solution: Type declaration management
// types/global.d.ts
declare global {
  interface Window {
    myGlobalVar: string;
  }
}

// Override conflicting types
declare module 'some-library' {
  export interface SomeInterface {
    newProperty: string;
  }
}
```

## Performance Problems

### React Performance Debugging
```typescript
// ✅ Performance debugging utilities
import { Profiler, ProfilerOnRenderCallback } from 'react';

const onRenderCallback: ProfilerOnRenderCallback = (
  id,
  phase,
  actualDuration,
  baseDuration,
  startTime,
  commitTime,
  interactions
) => {
  console.log('Profiler:', {
    id,
    phase,
    actualDuration,
    baseDuration,
    startTime,
    commitTime,
    interactions: Array.from(interactions)
  });
  
  // Log slow renders
  if (actualDuration > 16) {
    console.warn(`Slow render detected in ${id}: ${actualDuration}ms`);
  }
};

const MyApp = () => (
  <Profiler id="App" onRender={onRenderCallback}>
    <Header />
    <Main />
    <Footer />
  </Profiler>
);

// ✅ Custom hook for tracking renders
const useRenderCount = (componentName: string) => {
  const renderCount = useRef(0);
  
  useEffect(() => {
    renderCount.current += 1;
    console.log(`${componentName} render count:`, renderCount.current);
  });
  
  return renderCount.current;
};

// ✅ Memory leak detection
const useMemoryLeakDetection = () => {
  useEffect(() => {
    const checkMemory = () => {
      if (performance.memory) {
        const { usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit } = performance.memory;
        const memoryUsage = (usedJSHeapSize / jsHeapSizeLimit) * 100;
        
        if (memoryUsage > 80) {
          console.warn('High memory usage detected:', memoryUsage.toFixed(2) + '%');
        }
        
        console.log('Memory usage:', {
          used: Math.round(usedJSHeapSize / 1048576) + ' MB',
          total: Math.round(totalJSHeapSize / 1048576) + ' MB',
          limit: Math.round(jsHeapSizeLimit / 1048576) + ' MB',
          percentage: memoryUsage.toFixed(2) + '%'
        });
      }
    };
    
    const interval = setInterval(checkMemory, 5000);
    return () => clearInterval(interval);
  }, []);
};
```

### React Native Performance Issues
```typescript
// ❌ Problem: FlatList performance issues
<FlatList
  data={largeDataset}
  renderItem={({ item }) => (
    <ExpensiveItem item={item} /> // Re-renders on every scroll
  )}
/>

// ✅ Solution: Optimize FlatList
const renderItem = useCallback(({ item }) => (
  <MemoizedExpensiveItem item={item} />
), []);

const keyExtractor = useCallback((item) => item.id, []);

const getItemLayout = useCallback((data, index) => ({
  length: ITEM_HEIGHT,
  offset: ITEM_HEIGHT * index,
  index,
}), []);

<FlatList
  data={largeDataset}
  renderItem={renderItem}
  keyExtractor={keyExtractor}
  getItemLayout={getItemLayout}
  maxToRenderPerBatch={10}
  updateCellsBatchingPeriod={50}
  initialNumToRender={20}
  windowSize={21}
  removeClippedSubviews={true}
  onEndReachedThreshold={0.5}
/>

// ✅ Memoized list item
const MemoizedExpensiveItem = React.memo<{ item: Item }>(({ item }) => (
  <View>
    <Text>{item.title}</Text>
    <Text>{item.description}</Text>
  </View>
), (prevProps, nextProps) => {
  return prevProps.item.id === nextProps.item.id &&
         prevProps.item.updatedAt === nextProps.item.updatedAt;
});

// ❌ Problem: Image loading performance
<Image source={{ uri: largeImageUrl }} />

// ✅ Solution: Optimized image loading
const OptimizedImage = ({ uri, ...props }) => {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);

  return (
    <View style={styles.imageContainer}>
      <Image
        source={{ uri }}
        onLoad={() => setLoaded(true)}
        onError={() => setError(true)}
        style={[styles.image, { opacity: loaded ? 1 : 0 }]}
        resizeMode="cover"
        {...props}
      />
      {!loaded && !error && (
        <ActivityIndicator style={styles.loader} />
      )}
      {error && (
        <Text style={styles.errorText}>Failed to load</Text>
      )}
    </View>
  );
};

// ✅ Image caching with react-native-fast-image
import FastImage from 'react-native-fast-image';

<FastImage
  style={styles.image}
  source={{
    uri: imageUrl,
    priority: FastImage.priority.high,
    cache: FastImage.cacheControl.immutable,
  }}
  resizeMode={FastImage.resizeMode.cover}
/>
```

## Debugging Strategies

### React DevTools Setup
```typescript
// ✅ Enhanced debugging with component names
const MyComponent = function MyComponent(props) {
  // Component logic
};

// Better than:
const MyComponent = (props) => {
  // Component logic
};

// ✅ Custom DevTools integration
const useDevtools = (value: any, name: string) => {
  useEffect(() => {
    if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
      window.__REACT_DEVTOOLS_GLOBAL_HOOK__.onCommitFiberRoot(
        null,
        { memoizedState: value },
        null,
        false
      );
    }
  }, [value]);
};

// ✅ Debug hooks
const useDebugValue = (value: any, formatter?: (value: any) => any) => {
  React.useDebugValue(value, formatter);
  return value;
};

const useCounter = (initialValue = 0) => {
  const [count, setCount] = useState(initialValue);
  
  // This will show in React DevTools
  useDebugValue(count > 10 ? 'High' : 'Low');
  
  return [count, setCount];
};

// ✅ Component debugging wrapper
const withDebug = <P extends object>(
  Component: React.ComponentType<P>,
  debugName?: string
) => {
  const DebugWrapper = (props: P) => {
    const renderCount = useRef(0);
    renderCount.current++;
    
    useEffect(() => {
      console.log(`${debugName || Component.name} mounted`);
      return () => {
        console.log(`${debugName || Component.name} unmounted`);
      };
    }, []);
    
    useEffect(() => {
      console.log(`${debugName || Component.name} rendered (${renderCount.current})`);
    });
    
    return <Component {...props} />;
  };
  
  DebugWrapper.displayName = `withDebug(${Component.displayName || Component.name})`;
  return DebugWrapper;
};
```

### React Native Debugging
```typescript
// ✅ Network debugging
import { XMLHttpRequest } from 'react-native';

const originalSend = XMLHttpRequest.prototype.send;
XMLHttpRequest.prototype.send = function(body) {
  console.log('Network Request:', {
    method: this.method || 'GET',
    url: this.url,
    headers: this.requestHeaders,
    body: body
  });
  
  const originalOnReadyStateChange = this.onreadystatechange;
  this.onreadystatechange = function() {
    if (this.readyState === 4) {
      console.log('Network Response:', {
        status: this.status,
        statusText: this.statusText,
        responseText: this.responseText
      });
    }
    originalOnReadyStateChange?.call(this);
  };
  
  return originalSend.call(this, body);
};

// ✅ Layout debugging
import { UIManager } from 'react-native';

const logLayout = (node) => {
  UIManager.measure(node, (x, y, width, height, pageX, pageY) => {
    console.log('Layout:', { x, y, width, height, pageX, pageY });
  });
};

// ✅ Performance monitoring
import { Systrace } from 'react-native';

const performanceTrace = (name: string, fn: () => void) => {
  Systrace.beginEvent(name);
  try {
    return fn();
  } finally {
    Systrace.endEvent();
  }
};

// Usage
const expensiveOperation = () => {
  performanceTrace('ExpensiveOperation', () => {
    // Your expensive code here
  });
};

// ✅ Native module debugging
const debugNativeModule = (moduleName: string) => {
  const module = NativeModules[moduleName];
  if (!module) {
    console.error(`Native module ${moduleName} not found`);
    return;
  }
  
  console.log(`Native module ${moduleName} methods:`, Object.keys(module));
  
  // Wrap methods with logging
  Object.keys(module).forEach(methodName => {
    if (typeof module[methodName] === 'function') {
      const originalMethod = module[methodName];
      module[methodName] = (...args) => {
        console.log(`Calling ${moduleName}.${methodName}`, args);
        return originalMethod.apply(module, args);
      };
    }
  });
};
```

## Error Handling Patterns

### React Error Boundaries
```typescript
// ✅ Comprehensive error boundary
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string | null;
}

class ErrorBoundary extends React.Component<
  React.PropsWithChildren<{
    fallback?: React.ComponentType<{ error: Error; errorId: string; retry: () => void }>;
    onError?: (error: Error, errorInfo: ErrorInfo, errorId: string) => void;
  }>,
  ErrorBoundaryState
> {
  private retryTimeoutId: number | null = null;

  constructor(props: any) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    const errorId = Math.random().toString(36).substr(2, 9);
    return {
      hasError: true,
      error,
      errorId,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const errorId = this.state.errorId!;
    
    this.setState({ errorInfo });
    
    // Log error
    console.error('Error Boundary caught an error:', {
      error,
      errorInfo,
      errorId,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
    });
    
    // Report to error tracking service
    this.props.onError?.(error, errorInfo, errorId);
    
    // Auto-retry after 5 seconds
    this.retryTimeoutId = window.setTimeout(() => {
      this.handleRetry();
    }, 5000);
  }

  componentWillUnmount() {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
    });
  };

  render() {
    if (this.state.hasError) {
      const FallbackComponent = this.props.fallback || DefaultErrorFallback;
      return (
        <FallbackComponent
          error={this.state.error!}
          errorId={this.state.errorId!}
          retry={this.handleRetry}
        />
      );
    }

    return this.props.children;
  }
}

// ✅ Default error fallback component
const DefaultErrorFallback: React.FC<{
  error: Error;
  errorId: string;
  retry: () => void;
}> = ({ error, errorId, retry }) => (
  <div className="error-boundary">
    <h2>Something went wrong</h2>
    <details>
      <summary>Error details (ID: {errorId})</summary>
      <pre>{error.message}</pre>
      <pre>{error.stack}</pre>
    </details>
    <button onClick={retry}>Try again</button>
  </div>
);

// ✅ Hook-based error boundary
const useErrorHandler = () => {
  const [error, setError] = useState<Error | null>(null);

  const resetError = useCallback(() => {
    setError(null);
  }, []);

  const captureError = useCallback((error: Error) => {
    setError(error);
  }, []);

  useEffect(() => {
    if (error) {
      throw error;
    }
  }, [error]);

  return { captureError, resetError };
};

// Usage
const MyComponent = () => {
  const { captureError } = useErrorHandler();

  const handleAsyncError = async () => {
    try {
      await riskyAsyncOperation();
    } catch (error) {
      captureError(error);
    }
  };

  return <button onClick={handleAsyncError}>Do something risky</button>;
};
```

### Global Error Handling
```typescript
// ✅ Global error handlers
class GlobalErrorHandler {
  private static instance: GlobalErrorHandler;
  private errorReportingService: ErrorReportingService;

  constructor() {
    this.setupGlobalHandlers();
    this.errorReportingService = new ErrorReportingService();
  }

  static getInstance(): GlobalErrorHandler {
    if (!GlobalErrorHandler.instance) {
      GlobalErrorHandler.instance = new GlobalErrorHandler();
    }
    return GlobalErrorHandler.instance;
  }

  private setupGlobalHandlers() {
    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      console.error('Unhandled promise rejection:', event.reason);
      this.reportError(new Error(event.reason), 'unhandled-promise');
      event.preventDefault();
    });

    // Global JavaScript errors
    window.addEventListener('error', (event) => {
      console.error('Global error:', event.error);
      this.reportError(event.error, 'global-error');
    });

    // React errors (when not caught by error boundaries)
    const originalConsoleError = console.error;
    console.error = (...args) => {
      if (args[0]?.includes?.('React')) {
        this.reportError(new Error(args.join(' ')), 'react-error');
      }
      originalConsoleError.apply(console, args);
    };
  }

  reportError(error: Error, context: string, additionalData?: any) {
    const errorReport = {
      message: error.message,
      stack: error.stack,
      context,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      additionalData,
    };

    this.errorReportingService.send(errorReport);
  }
}

// Initialize global error handler
GlobalErrorHandler.getInstance();

// ✅ Network error handling
const createApiClient = () => {
  const client = axios.create({
    baseURL: process.env.REACT_APP_API_URL,
    timeout: 10000,
  });

  client.interceptors.response.use(
    (response) => response,
    (error) => {
      const errorMessage = error.response?.data?.message || error.message;
      const statusCode = error.response?.status;

      console.error('API Error:', {
        message: errorMessage,
        status: statusCode,
        url: error.config?.url,
        method: error.config?.method,
      });

      // Report to error tracking
      GlobalErrorHandler.getInstance().reportError(
        new Error(`API Error: ${errorMessage}`),
        'api-error',
        { statusCode, url: error.config?.url }
      );

      return Promise.reject(error);
    }
  );

  return client;
};
```

---

*This completes the comprehensive React and React Native documentation series covering all concepts from basic to advanced levels, including troubleshooting and best practices.*
