# React Native Advanced Concepts

## Table of Contents
- [Native Modules Development](#native-modules-development)
- [Bridge Communication](#bridge-communication)
- [Turbo Modules (New Architecture)](#turbo-modules-new-architecture)
- [Fabric Renderer (New Architecture)](#fabric-renderer-new-architecture)
- [Advanced Performance Profiling](#advanced-performance-profiling)
- [Memory Management](#memory-management)
- [Custom Native Components](#custom-native-components)
- [Threading and Concurrency](#threading-and-concurrency)
- [Advanced Debugging](#advanced-debugging)
- [Architecture Patterns](#architecture-patterns)

## Native Modules Development

### iOS Native Module
```objc
// ios/RNCustomModule.h
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RNCustomModule : RCTEventEmitter <RCTBridgeModule>
@end

// ios/RNCustomModule.m
#import "RNCustomModule.h"
#import <React/RCTLog.h>

@implementation RNCustomModule

RCT_EXPORT_MODULE();

// Export methods
RCT_EXPORT_METHOD(calculateSum:(double)a 
                  withB:(double)b 
                  resolver:(RCTPromiseResolveBlock)resolve 
                  rejecter:(RCTPromiseRejectBlock)reject) {
    
    double sum = a + b;
    resolve(@(sum));
}

RCT_EXPORT_METHOD(processDataWithCallback:(NSArray *)data 
                  callback:(RCTResponseSenderBlock)callback) {
    
    NSMutableArray *processedData = [[NSMutableArray alloc] init];
    
    for (NSNumber *number in data) {
        [processedData addObject:@([number doubleValue] * 2)];
    }
    
    callback(@[[NSNull null], processedData]);
}

// Export constants
- (NSDictionary *)constantsToExport {
    return @{
        @"API_URL": @"https://api.example.com",
        @"VERSION": @"1.0.0",
        @"PLATFORM": @"iOS"
    };
}

// Event emitter methods
- (NSArray<NSString *> *)supportedEvents {
    return @[@"DataReceived", @"ErrorOccurred"];
}

RCT_EXPORT_METHOD(startListening) {
    // Start some background process
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        // Simulate data processing
        [NSThread sleepForTimeInterval:2.0];
        
        dispatch_async(dispatch_get_main_queue(), ^{
            [self sendEventWithName:@"DataReceived" 
                               body:@{@"data": @"Processed data"}];
        });
    });
}

// Require main queue setup
+ (BOOL)requiresMainQueueSetup {
    return YES;
}

@end
```

### Android Native Module
```java
// android/app/src/main/java/com/yourapp/CustomModule.java
package com.yourapp;

import android.os.Handler;
import android.os.Looper;

import com.facebook.react.bridge.*;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import java.util.HashMap;
import java.util.Map;

public class CustomModule extends ReactContextBaseJavaModule {
    
    private ReactApplicationContext reactContext;
    
    public CustomModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
    }
    
    @Override
    public String getName() {
        return "CustomModule";
    }
    
    // Export constants
    @Override
    public Map<String, Object> getConstants() {
        final Map<String, Object> constants = new HashMap<>();
        constants.put("API_URL", "https://api.example.com");
        constants.put("VERSION", "1.0.0");
        constants.put("PLATFORM", "Android");
        return constants;
    }
    
    @ReactMethod
    public void calculateSum(double a, double b, Promise promise) {
        try {
            double sum = a + b;
            promise.resolve(sum);
        } catch (Exception e) {
            promise.reject("CALCULATION_ERROR", e.getMessage(), e);
        }
    }
    
    @ReactMethod
    public void processDataWithCallback(ReadableArray data, Callback callback) {
        try {
            WritableArray processedData = Arguments.createArray();
            
            for (int i = 0; i < data.size(); i++) {
                double value = data.getDouble(i);
                processedData.pushDouble(value * 2);
            }
            
            callback.invoke(null, processedData);
        } catch (Exception e) {
            callback.invoke(e.getMessage(), null);
        }
    }
    
    @ReactMethod
    public void startListening() {
        // Simulate background processing
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(2000);
                    
                    // Send event back to React Native
                    Handler mainHandler = new Handler(Looper.getMainLooper());
                    mainHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            WritableMap params = Arguments.createMap();
                            params.putString("data", "Processed data");
                            sendEvent("DataReceived", params);
                        }
                    });
                } catch (InterruptedException e) {
                    sendEvent("ErrorOccurred", null);
                }
            }
        }).start();
    }
    
    private void sendEvent(String eventName, WritableMap params) {
        if (reactContext.hasActiveCatalystInstance()) {
            reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
                .emit(eventName, params);
        }
    }
}

// android/app/src/main/java/com/yourapp/CustomPackage.java
package com.yourapp;

import com.facebook.react.ReactPackage;
import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.uimanager.ViewManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CustomPackage implements ReactPackage {
    
    @Override
    public List<ViewManager> createViewManagers(ReactApplicationContext reactContext) {
        return Collections.emptyList();
    }
    
    @Override
    public List<NativeModule> createNativeModules(ReactApplicationContext reactContext) {
        List<NativeModule> modules = new ArrayList<>();
        modules.add(new CustomModule(reactContext));
        return modules;
    }
}
```

### JavaScript Interface
```javascript
// NativeModules.js
import { NativeModules, NativeEventEmitter } from 'react-native';

const { CustomModule } = NativeModules;

class CustomNativeModule {
  constructor() {
    this.eventEmitter = new NativeEventEmitter(CustomModule);
    this.listeners = new Map();
  }

  // Constants
  get constants() {
    return CustomModule.getConstants();
  }

  // Promise-based method
  async calculateSum(a, b) {
    try {
      const result = await CustomModule.calculateSum(a, b);
      return result;
    } catch (error) {
      throw new Error(`Calculation failed: ${error.message}`);
    }
  }

  // Callback-based method
  processData(data, callback) {
    CustomModule.processDataWithCallback(data, (error, result) => {
      if (error) {
        callback(new Error(error), null);
      } else {
        callback(null, result);
      }
    });
  }

  // Promise wrapper for callback method
  processDataAsync(data) {
    return new Promise((resolve, reject) => {
      this.processData(data, (error, result) => {
        if (error) {
          reject(error);
        } else {
          resolve(result);
        }
      });
    });
  }

  // Event listeners
  addEventListener(eventName, listener) {
    const subscription = this.eventEmitter.addListener(eventName, listener);
    this.listeners.set(listener, subscription);
    return subscription;
  }

  removeEventListener(eventName, listener) {
    const subscription = this.listeners.get(listener);
    if (subscription) {
      subscription.remove();
      this.listeners.delete(listener);
    }
  }

  removeAllListeners(eventName) {
    this.eventEmitter.removeAllListeners(eventName);
    this.listeners.clear();
  }

  // Start background processing
  startListening() {
    CustomModule.startListening();
  }
}

export default new CustomNativeModule();

// Usage example
import CustomNativeModule from './NativeModules';

const ExampleComponent = () => {
  useEffect(() => {
    // Listen for events
    const dataListener = CustomNativeModule.addEventListener('DataReceived', (data) => {
      console.log('Received data:', data);
    });

    const errorListener = CustomNativeModule.addEventListener('ErrorOccurred', (error) => {
      console.error('Native module error:', error);
    });

    // Start listening
    CustomNativeModule.startListening();

    return () => {
      CustomNativeModule.removeEventListener('DataReceived', dataListener);
      CustomNativeModule.removeEventListener('ErrorOccurred', errorListener);
    };
  }, []);

  const handleCalculation = async () => {
    try {
      const result = await CustomNativeModule.calculateSum(10, 20);
      console.log('Sum:', result);
    } catch (error) {
      console.error('Calculation error:', error);
    }
  };

  const handleDataProcessing = async () => {
    try {
      const data = [1, 2, 3, 4, 5];
      const processed = await CustomNativeModule.processDataAsync(data);
      console.log('Processed data:', processed);
    } catch (error) {
      console.error('Processing error:', error);
    }
  };

  return (
    <View>
      <Button title="Calculate Sum" onPress={handleCalculation} />
      <Button title="Process Data" onPress={handleDataProcessing} />
    </View>
  );
};
```

## Bridge Communication

### Understanding the Bridge
```javascript
// Bridge communication patterns
class BridgeManager {
  static batching = {
    // Batch multiple calls
    batchCalls: [],
    timeout: null,
    
    addCall(moduleName, methodName, args) {
      this.batchCalls.push({ moduleName, methodName, args });
      
      if (!this.timeout) {
        this.timeout = setTimeout(() => {
          this.executeBatch();
        }, 16); // Next frame
      }
    },
    
    executeBatch() {
      const calls = [...this.batchCalls];
      this.batchCalls = [];
      this.timeout = null;
      
      // Execute all calls in a single bridge transaction
      calls.forEach(({ moduleName, methodName, args }) => {
        NativeModules[moduleName][methodName](...args);
      });
    }
  };

  // Optimize bridge calls
  static optimizeCalls = {
    // Debounce frequent calls
    debounced: new Map(),
    
    debounce(key, fn, delay = 100) {
      if (this.debounced.has(key)) {
        clearTimeout(this.debounced.get(key));
      }
      
      const timeoutId = setTimeout(fn, delay);
      this.debounced.set(key, timeoutId);
    },
    
    // Throttle high-frequency calls
    throttled: new Map(),
    
    throttle(key, fn, limit = 100) {
      if (this.throttled.has(key)) {
        return;
      }
      
      fn();
      this.throttled.set(key, true);
      
      setTimeout(() => {
        this.throttled.delete(key);
      }, limit);
    }
  };

  // Monitor bridge traffic
  static monitoring = {
    calls: [],
    maxCalls: 1000,
    
    trackCall(moduleName, methodName, args, timestamp = Date.now()) {
      this.calls.push({ moduleName, methodName, args, timestamp });
      
      if (this.calls.length > this.maxCalls) {
        this.calls = this.calls.slice(-this.maxCalls);
      }
    },
    
    getStats() {
      const now = Date.now();
      const recentCalls = this.calls.filter(call => now - call.timestamp < 5000);
      
      const moduleStats = recentCalls.reduce((stats, call) => {
        const key = `${call.moduleName}.${call.methodName}`;
        stats[key] = (stats[key] || 0) + 1;
        return stats;
      }, {});
      
      return {
        totalCalls: this.calls.length,
        recentCalls: recentCalls.length,
        callsPerSecond: recentCalls.length / 5,
        moduleStats
      };
    }
  };
}

// Bridge performance optimization
const BridgeOptimizer = {
  // Reduce serialization overhead
  optimizeData(data) {
    if (typeof data === 'object' && data !== null) {
      // Remove unnecessary properties
      const optimized = {};
      Object.keys(data).forEach(key => {
        if (data[key] !== undefined && data[key] !== null) {
          optimized[key] = data[key];
        }
      });
      return optimized;
    }
    return data;
  },

  // Use MessageQueue for high-frequency updates
  setupMessageQueue() {
    const MessageQueue = require('react-native/Libraries/BatchedBridge/MessageQueue');
    
    const originalCallNativeSyncHook = MessageQueue.prototype.__callNativeSyncHook;
    MessageQueue.prototype.__callNativeSyncHook = function(moduleID, methodID, params) {
      // Add custom logging or optimization here
      BridgeManager.monitoring.trackCall(moduleID, methodID, params);
      return originalCallNativeSyncHook.call(this, moduleID, methodID, params);
    };
  },

  // Minimize bridge crossings
  batchOperations(operations) {
    return new Promise((resolve, reject) => {
      const batchId = Date.now();
      const results = [];
      
      operations.forEach((operation, index) => {
        operation()
          .then(result => {
            results[index] = { success: true, data: result };
            if (results.filter(r => r !== undefined).length === operations.length) {
              resolve(results);
            }
          })
          .catch(error => {
            results[index] = { success: false, error };
            if (results.filter(r => r !== undefined).length === operations.length) {
              resolve(results);
            }
          });
      });
    });
  }
};
```

## Turbo Modules (New Architecture)

### Turbo Module Specification
```typescript
// TurboModule interface definition
import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  // Synchronous methods
  getConstants(): {
    API_URL: string;
    VERSION: string;
  };
  
  // Asynchronous methods
  calculateSum(a: number, b: number): Promise<number>;
  processData(data: Array<number>): Promise<Array<number>>;
  
  // With callbacks
  fetchUserData(userId: string, callback: (error: string | null, data: Object | null) => void): void;
  
  // Event emitter methods
  addListener(eventName: string): void;
  removeListeners(count: number): void;
}

export default TurboModuleRegistry.getEnforcing<Spec>('CustomTurboModule');
```

### iOS Turbo Module Implementation
```objc
// ios/CustomTurboModule.h
#import <React/RCTBridgeModule.h>
#import <ReactCommon/RCTTurboModule.h>

@interface CustomTurboModule : NSObject <RCTBridgeModule, RCTTurboModule>
@end

// ios/CustomTurboModule.mm
#import "CustomTurboModule.h"
#import <React/RCTUtils.h>

@implementation CustomTurboModule

RCT_EXPORT_MODULE();

// Synchronous method
- (facebook::react::ModuleConstants<JS::NativeCustomTurboModule::Constants>)constantsToExport {
    return facebook::react::typedConstants<JS::NativeCustomTurboModule::Constants>({
        .API_URL = @"https://api.example.com",
        .VERSION = @"1.0.0"
    });
}

// Asynchronous method
- (void)calculateSum:(double)a 
                   b:(double)b 
             resolve:(RCTPromiseResolveBlock)resolve 
              reject:(RCTPromiseRejectBlock)reject {
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        double result = a + b;
        dispatch_async(dispatch_get_main_queue(), ^{
            resolve(@(result));
        });
    });
}

- (void)processData:(NSArray<NSNumber *> *)data 
            resolve:(RCTPromiseResolveBlock)resolve 
             reject:(RCTPromiseRejectBlock)reject {
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        NSMutableArray *processedData = [[NSMutableArray alloc] init];
        
        for (NSNumber *number in data) {
            [processedData addObject:@([number doubleValue] * 2)];
        }
        
        dispatch_async(dispatch_get_main_queue(), ^{
            resolve(processedData);
        });
    });
}

// Callback method
- (void)fetchUserData:(NSString *)userId 
             callback:(RCTResponseSenderBlock)callback {
    
    // Simulate API call
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        // Simulate delay
        [NSThread sleepForTimeInterval:1.0];
        
        NSDictionary *userData = @{
            @"id": userId,
            @"name": @"John Doe",
            @"email": @"john@example.com"
        };
        
        dispatch_async(dispatch_get_main_queue(), ^{
            callback(@[[NSNull null], userData]);
        });
    });
}

// Standard methods for TurboModule
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params {
    return std::make_shared<facebook::react::NativeCustomTurboModuleSpecJSI>(params);
}

@end
```

### Android Turbo Module Implementation
```java
// android/app/src/main/java/com/yourapp/CustomTurboModule.java
package com.yourapp;

import androidx.annotation.NonNull;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Callback;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CustomTurboModule extends NativeCustomTurboModuleSpec {
    
    private final ExecutorService executor = Executors.newCachedThreadPool();
    
    public CustomTurboModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }
    
    @Override
    @NonNull
    public String getName() {
        return "CustomTurboModule";
    }
    
    @Override
    public Map<String, Object> getTypedExportedConstants() {
        Map<String, Object> constants = new HashMap<>();
        constants.put("API_URL", "https://api.example.com");
        constants.put("VERSION", "1.0.0");
        return constants;
    }
    
    @Override
    public void calculateSum(double a, double b, Promise promise) {
        executor.execute(() -> {
            try {
                double result = a + b;
                promise.resolve(result);
            } catch (Exception e) {
                promise.reject("CALCULATION_ERROR", e.getMessage(), e);
            }
        });
    }
    
    @Override
    public void processData(ReadableArray data, Promise promise) {
        executor.execute(() -> {
            try {
                WritableArray processedData = Arguments.createArray();
                
                for (int i = 0; i < data.size(); i++) {
                    double value = data.getDouble(i);
                    processedData.pushDouble(value * 2);
                }
                
                promise.resolve(processedData);
            } catch (Exception e) {
                promise.reject("PROCESSING_ERROR", e.getMessage(), e);
            }
        });
    }
    
    @Override
    public void fetchUserData(String userId, Callback callback) {
        executor.execute(() -> {
            try {
                // Simulate API call delay
                Thread.sleep(1000);
                
                WritableMap userData = Arguments.createMap();
                userData.putString("id", userId);
                userData.putString("name", "John Doe");
                userData.putString("email", "john@example.com");
                
                callback.invoke(null, userData);
            } catch (Exception e) {
                callback.invoke(e.getMessage(), null);
            }
        });
    }
}
```

### Turbo Module Registration
```javascript
// js/CustomTurboModule.js
import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  getConstants(): {
    API_URL: string;
    VERSION: string;
  };
  calculateSum(a: number, b: number): Promise<number>;
  processData(data: Array<number>): Promise<Array<number>>;
  fetchUserData(userId: string, callback: (error: string | null, data: Object | null) => void): void;
}

export default TurboModuleRegistry.getEnforcing<Spec>('CustomTurboModule');

// Usage
import CustomTurboModule from './CustomTurboModule';

const useTurboModule = () => {
  const [result, setResult] = useState(null);
  
  const testTurboModule = async () => {
    try {
      // Get constants
      const constants = CustomTurboModule.getConstants();
      console.log('Constants:', constants);
      
      // Calculate sum
      const sum = await CustomTurboModule.calculateSum(10, 20);
      console.log('Sum:', sum);
      
      // Process data
      const processed = await CustomTurboModule.processData([1, 2, 3, 4, 5]);
      console.log('Processed:', processed);
      
      // Fetch user data with callback
      CustomTurboModule.fetchUserData('123', (error, data) => {
        if (error) {
          console.error('Error:', error);
        } else {
          console.log('User data:', data);
          setResult(data);
        }
      });
    } catch (error) {
      console.error('Turbo module error:', error);
    }
  };
  
  return { result, testTurboModule };
};
```

## Fabric Renderer (New Architecture)

### Custom Fabric Component
```typescript
// js/CustomViewNativeComponent.ts
import type { ViewProps } from 'react-native';
import type { HostComponent } from 'react-native';
import codegenNativeComponent from 'react-native/Libraries/Utilities/codegenNativeComponent';

export interface NativeProps extends ViewProps {
  // Custom properties
  title?: string;
  subtitle?: string;
  backgroundColor?: string;
  isEnabled?: boolean;
  onCustomEvent?: (event: {
    nativeEvent: {
      value: string;
      timestamp: number;
    };
  }) => void;
}

export default codegenNativeComponent<NativeProps>('CustomView') as HostComponent<NativeProps>;
```

### iOS Fabric Component Implementation
```objc
// ios/CustomViewComponentView.h
#import <React/RCTViewComponentView.h>
#import <React/RCTComponentViewProtocol.h>

@interface CustomViewComponentView : RCTViewComponentView
@end

// ios/CustomViewComponentView.mm
#import "CustomViewComponentView.h"
#import <React/RCTConversions.h>
#import <React/RCTFabricComponentsPlugins.h>

@implementation CustomViewComponentView {
    UILabel *_titleLabel;
    UILabel *_subtitleLabel;
    UIView *_containerView;
}

- (instancetype)initWithFrame:(CGRect)frame {
    if (self = [super initWithFrame:frame]) {
        [self setupViews];
    }
    return self;
}

- (void)setupViews {
    _containerView = [[UIView alloc] init];
    _containerView.backgroundColor = [UIColor lightGrayColor];
    [self addSubview:_containerView];
    
    _titleLabel = [[UILabel alloc] init];
    _titleLabel.font = [UIFont boldSystemFontOfSize:18];
    _titleLabel.textColor = [UIColor blackColor];
    [_containerView addSubview:_titleLabel];
    
    _subtitleLabel = [[UILabel alloc] init];
    _subtitleLabel.font = [UIFont systemFontOfSize:14];
    _subtitleLabel.textColor = [UIColor grayColor];
    [_containerView addSubview:_subtitleLabel];
    
    // Add tap gesture
    UITapGestureRecognizer *tapGesture = [[UITapGestureRecognizer alloc] 
                                         initWithTarget:self 
                                         action:@selector(handleTap:)];
    [_containerView addGestureRecognizer:tapGesture];
}

- (void)updateProps:(Props::Shared const &)props 
           oldProps:(Props::Shared const &)oldProps {
    
    const auto &oldCustomProps = *std::static_pointer_cast<CustomViewProps const>(oldProps);
    const auto &newCustomProps = *std::static_pointer_cast<CustomViewProps const>(props);
    
    // Update title
    if (oldCustomProps.title != newCustomProps.title) {
        _titleLabel.text = RCTNSStringFromString(newCustomProps.title);
    }
    
    // Update subtitle
    if (oldCustomProps.subtitle != newCustomProps.subtitle) {
        _subtitleLabel.text = RCTNSStringFromString(newCustomProps.subtitle);
    }
    
    // Update background color
    if (oldCustomProps.backgroundColor != newCustomProps.backgroundColor) {
        _containerView.backgroundColor = RCTUIColorFromSharedColor(newCustomProps.backgroundColor);
    }
    
    // Update enabled state
    if (oldCustomProps.isEnabled != newCustomProps.isEnabled) {
        _containerView.alpha = newCustomProps.isEnabled ? 1.0 : 0.5;
        _containerView.userInteractionEnabled = newCustomProps.isEnabled;
    }
    
    [super updateProps:props oldProps:oldProps];
}

- (void)layoutSubviews {
    [super layoutSubviews];
    
    CGRect bounds = self.bounds;
    _containerView.frame = bounds;
    
    CGFloat padding = 16;
    CGFloat titleHeight = 24;
    CGFloat subtitleHeight = 20;
    
    _titleLabel.frame = CGRectMake(padding, padding, 
                                  bounds.size.width - 2 * padding, titleHeight);
    _subtitleLabel.frame = CGRectMake(padding, padding + titleHeight + 8, 
                                     bounds.size.width - 2 * padding, subtitleHeight);
}

- (void)handleTap:(UITapGestureRecognizer *)gesture {
    if (_eventEmitter) {
        auto customEventEmitter = std::static_pointer_cast<CustomViewEventEmitter const>(_eventEmitter);
        
        CustomViewEventEmitter::OnCustomEvent event;
        event.value = "tapped";
        event.timestamp = [[NSDate date] timeIntervalSince1970] * 1000;
        
        customEventEmitter->onCustomEvent(event);
    }
}

+ (ComponentDescriptorProvider)componentDescriptorProvider {
    return concreteComponentDescriptorProvider<CustomViewComponentDescriptor>();
}

@end

Class<RCTComponentViewProtocol> CustomViewCls(void) {
    return CustomViewComponentView.class;
}
```

### Android Fabric Component Implementation
```java
// android/app/src/main/java/com/yourapp/CustomViewManager.java
package com.yourapp;

import android.graphics.Color;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.common.MapBuilder;
import com.facebook.react.uimanager.SimpleViewManager;
import com.facebook.react.uimanager.ThemedReactContext;
import com.facebook.react.uimanager.UIManagerHelper;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.uimanager.events.Event;
import com.facebook.react.uimanager.events.EventDispatcher;

import java.util.Map;

public class CustomViewManager extends SimpleViewManager<LinearLayout> {
    
    public static final String REACT_CLASS = "CustomView";
    private static final int COMMAND_SET_TITLE = 1;
    
    @Override
    @NonNull
    public String getName() {
        return REACT_CLASS;
    }
    
    @Override
    @NonNull
    public LinearLayout createViewInstance(@NonNull ThemedReactContext reactContext) {
        LinearLayout layout = new LinearLayout(reactContext);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(48, 48, 48, 48);
        
        // Create title
        TextView titleView = new TextView(reactContext);
        titleView.setId(R.id.title);
        titleView.setTextSize(18);
        titleView.setTextColor(Color.BLACK);
        layout.addView(titleView);
        
        // Create subtitle
        TextView subtitleView = new TextView(reactContext);
        subtitleView.setId(R.id.subtitle);
        subtitleView.setTextSize(14);
        subtitleView.setTextColor(Color.GRAY);
        layout.addView(subtitleView);
        
        // Add click listener
        layout.setOnClickListener(v -> {
            WritableMap event = Arguments.createMap();
            event.putString("value", "tapped");
            event.putDouble("timestamp", System.currentTimeMillis());
            
            EventDispatcher eventDispatcher = UIManagerHelper.getEventDispatcherForReactTag(
                reactContext, layout.getId());
            if (eventDispatcher != null) {
                eventDispatcher.dispatchEvent(new CustomEvent(layout.getId(), event));
            }
        });
        
        return layout;
    }
    
    @ReactProp(name = "title")
    public void setTitle(LinearLayout view, @Nullable String title) {
        TextView titleView = view.findViewById(R.id.title);
        if (titleView != null) {
            titleView.setText(title);
        }
    }
    
    @ReactProp(name = "subtitle")
    public void setSubtitle(LinearLayout view, @Nullable String subtitle) {
        TextView subtitleView = view.findViewById(R.id.subtitle);
        if (subtitleView != null) {
            subtitleView.setText(subtitle);
        }
    }
    
    @ReactProp(name = "backgroundColor", customType = "Color")
    public void setBackgroundColor(LinearLayout view, @Nullable Integer backgroundColor) {
        if (backgroundColor != null) {
            view.setBackgroundColor(backgroundColor);
        }
    }
    
    @ReactProp(name = "isEnabled", defaultBoolean = true)
    public void setEnabled(LinearLayout view, boolean enabled) {
        view.setEnabled(enabled);
        view.setAlpha(enabled ? 1.0f : 0.5f);
    }
    
    @Override
    public Map<String, Object> getExportedCustomDirectEventTypeConstants() {
        return MapBuilder.<String, Object>builder()
            .put("onCustomEvent", MapBuilder.of("registrationName", "onCustomEvent"))
            .build();
    }
    
    // Custom event class
    static class CustomEvent extends Event<CustomEvent> {
        private final WritableMap eventData;
        
        public CustomEvent(int viewId, WritableMap eventData) {
            super(viewId);
            this.eventData = eventData;
        }
        
        @Override
        public String getEventName() {
            return "onCustomEvent";
        }
        
        @Override
        protected WritableMap getEventData() {
            return eventData;
        }
    }
}
```

### Using Custom Fabric Component
```javascript
// CustomView.js
import React from 'react';
import CustomViewNativeComponent from './CustomViewNativeComponent';

const CustomView = ({ 
  title, 
  subtitle, 
  backgroundColor = '#f0f0f0',
  isEnabled = true,
  onCustomEvent,
  style,
  ...props 
}) => {
  const handleCustomEvent = (event) => {
    console.log('Custom event received:', event.nativeEvent);
    onCustomEvent?.(event);
  };

  return (
    <CustomViewNativeComponent
      style={[{ height: 120 }, style]}
      title={title}
      subtitle={subtitle}
      backgroundColor={backgroundColor}
      isEnabled={isEnabled}
      onCustomEvent={handleCustomEvent}
      {...props}
    />
  );
};

export default CustomView;

// Usage
const App = () => {
  const handleEvent = (event) => {
    console.log('Event data:', event.nativeEvent);
  };

  return (
    <View style={{ flex: 1, padding: 20 }}>
      <CustomView
        title="Custom Native Component"
        subtitle="Built with Fabric renderer"
        backgroundColor="#e3f2fd"
        isEnabled={true}
        onCustomEvent={handleEvent}
        style={{ marginVertical: 10 }}
      />
    </View>
  );
};
```

## Advanced Performance Profiling

### Performance Monitoring Suite
```javascript
// PerformanceProfiler.js
import { InteractionManager, Systrace, PerfMonitor } from 'react-native';

class PerformanceProfiler {
  constructor() {
    this.metrics = {
      renders: [],
      interactions: [],
      bridgeCalls: [],
      memoryUsage: [],
      jsCallHistory: []
    };
    
    this.isEnabled = __DEV__;
    this.setupProfiling();
  }

  setupProfiling() {
    if (!this.isEnabled) return;

    // Monitor renders
    this.monitorRenders();
    
    // Monitor interactions
    this.monitorInteractions();
    
    // Monitor bridge calls
    this.monitorBridgeCalls();
    
    // Monitor memory usage
    this.monitorMemoryUsage();
    
    // Monitor JavaScript thread
    this.monitorJSThread();
  }

  monitorRenders() {
    const originalCreateElement = React.createElement;
    React.createElement = (...args) => {
      const startTime = performance.now();
      const element = originalCreateElement.apply(React, args);
      const endTime = performance.now();
      
      this.metrics.renders.push({
        component: args[0]?.name || 'Unknown',
        duration: endTime - startTime,
        timestamp: Date.now()
      });
      
      return element;
    };
  }

  monitorInteractions() {
    InteractionManager.addInteractionCompleteListener(() => {
      this.metrics.interactions.push({
        timestamp: Date.now(),
        type: 'interaction_complete'
      });
    });
  }

  monitorBridgeCalls() {
    const MessageQueue = require('react-native/Libraries/BatchedBridge/MessageQueue');
    const originalCallNativeSyncHook = MessageQueue.prototype.__callNativeSyncHook;
    
    MessageQueue.prototype.__callNativeSyncHook = function(moduleID, methodID, params) {
      const startTime = performance.now();
      const result = originalCallNativeSyncHook.call(this, moduleID, methodID, params);
      const endTime = performance.now();
      
      this.metrics.bridgeCalls.push({
        moduleID,
        methodID,
        duration: endTime - startTime,
        timestamp: Date.now()
      });
      
      return result;
    }.bind(this);
  }

  monitorMemoryUsage() {
    setInterval(() => {
      if (performance.memory) {
        this.metrics.memoryUsage.push({
          used: performance.memory.usedJSHeapSize,
          total: performance.memory.totalJSHeapSize,
          limit: performance.memory.jsHeapSizeLimit,
          timestamp: Date.now()
        });
      }
    }, 1000);
  }

  monitorJSThread() {
    let lastTimestamp = performance.now();
    
    const checkJSThread = () => {
      const currentTimestamp = performance.now();
      const frameDuration = currentTimestamp - lastTimestamp;
      
      this.metrics.jsCallHistory.push({
        frameDuration,
        isBlocked: frameDuration > 16.67, // 60 FPS threshold
        timestamp: Date.now()
      });
      
      lastTimestamp = currentTimestamp;
      requestAnimationFrame(checkJSThread);
    };
    
    requestAnimationFrame(checkJSThread);
  }

  // Performance analysis methods
  analyzeRenderPerformance() {
    const recentRenders = this.metrics.renders
      .filter(render => Date.now() - render.timestamp < 5000)
      .sort((a, b) => b.duration - a.duration);
    
    return {
      slowestRenders: recentRenders.slice(0, 10),
      averageRenderTime: recentRenders.reduce((sum, r) => sum + r.duration, 0) / recentRenders.length,
      totalRenders: recentRenders.length
    };
  }

  analyzeBridgePerformance() {
    const recentCalls = this.metrics.bridgeCalls
      .filter(call => Date.now() - call.timestamp < 5000);
    
    const moduleStats = recentCalls.reduce((stats, call) => {
      const key = `${call.moduleID}.${call.methodID}`;
      if (!stats[key]) {
        stats[key] = { count: 0, totalDuration: 0 };
      }
      stats[key].count++;
      stats[key].totalDuration += call.duration;
      return stats;
    }, {});
    
    return {
      totalCalls: recentCalls.length,
      callsPerSecond: recentCalls.length / 5,
      moduleStats: Object.entries(moduleStats).map(([module, stats]) => ({
        module,
        count: stats.count,
        averageDuration: stats.totalDuration / stats.count
      })).sort((a, b) => b.averageDuration - a.averageDuration)
    };
  }

  analyzeJSThreadPerformance() {
    const recentFrames = this.metrics.jsCallHistory
      .filter(frame => Date.now() - frame.timestamp < 5000);
    
    const blockedFrames = recentFrames.filter(frame => frame.isBlocked);
    
    return {
      totalFrames: recentFrames.length,
      blockedFrames: blockedFrames.length,
      blockedPercentage: (blockedFrames.length / recentFrames.length) * 100,
      averageFrameDuration: recentFrames.reduce((sum, f) => sum + f.frameDuration, 0) / recentFrames.length,
      fps: 1000 / (recentFrames.reduce((sum, f) => sum + f.frameDuration, 0) / recentFrames.length)
    };
  }

  getPerformanceReport() {
    return {
      timestamp: Date.now(),
      renders: this.analyzeRenderPerformance(),
      bridge: this.analyzeBridgePerformance(),
      jsThread: this.analyzeJSThreadPerformance(),
      memory: this.metrics.memoryUsage.slice(-5)
    };
  }

  // Start/stop profiling
  startProfiling() {
    this.isEnabled = true;
    console.log('🔍 Performance profiling started');
  }

  stopProfiling() {
    this.isEnabled = false;
    console.log('⏹️ Performance profiling stopped');
    return this.getPerformanceReport();
  }

  // Clear metrics
  clearMetrics() {
    this.metrics = {
      renders: [],
      interactions: [],
      bridgeCalls: [],
      memoryUsage: [],
      jsCallHistory: []
    };
  }
}

export default new PerformanceProfiler();

// Usage
import PerformanceProfiler from './PerformanceProfiler';

const App = () => {
  useEffect(() => {
    // Start profiling in development
    if (__DEV__) {
      PerformanceProfiler.startProfiling();
      
      // Log performance report every 10 seconds
      const interval = setInterval(() => {
        const report = PerformanceProfiler.getPerformanceReport();
        console.log('Performance Report:', report);
      }, 10000);
      
      return () => {
        clearInterval(interval);
        PerformanceProfiler.stopProfiling();
      };
    }
  }, []);

  return <YourAppContent />;
};
```

---

*Continue to: [18-shared-patterns.md](./18-shared-patterns.md)*
