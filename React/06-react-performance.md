# React Performance Optimization

## Table of Contents
- [Understanding React Performance](#understanding-react-performance)
- [React.memo and PureComponent](#reactmemo-and-purecomponent)
- [useMemo and useCallback](#usememo-and-usecallback)
- [Code Splitting and Lazy Loading](#code-splitting-and-lazy-loading)
- [Virtual Scrolling](#virtual-scrolling)
- [Bundle Optimization](#bundle-optimization)
- [Profiling and Debugging](#profiling-and-debugging)
- [Performance Patterns](#performance-patterns)

## Understanding React Performance

### React Rendering Process
```javascript
// React's rendering phases:
// 1. Trigger - State change, props change, or parent re-render
// 2. Render - Create new virtual DOM tree
// 3. Commit - Apply changes to actual DOM

// Performance bottlenecks often occur in:
// - Unnecessary re-renders
// - Heavy computations during render
// - Large component trees
// - Inefficient reconciliation

import React, { useState, useEffect } from 'react';

function PerformanceExample() {
  const [count, setCount] = useState(0);
  const [users, setUsers] = useState([]);

  // This will re-run on every render - EXPENSIVE!
  const expensiveValue = users.map(user => ({
    ...user,
    fullName: `${user.firstName} ${user.lastName}`,
    isActive: user.lastLogin > Date.now() - 86400000
  }));

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      {/* This causes unnecessary re-renders */}
      <UserList users={expensiveValue} />
    </div>
  );
}
```

### Identifying Performance Issues
```javascript
import React, { Profiler } from 'react';

function onRenderCallback(id, phase, actualDuration, baseDuration, startTime, commitTime) {
  console.log('Component:', id);
  console.log('Phase:', phase); // 'mount' or 'update'
  console.log('Actual duration:', actualDuration);
  console.log('Base duration:', baseDuration);
  console.log('Start time:', startTime);
  console.log('Commit time:', commitTime);
  
  // Log slow components
  if (actualDuration > 10) {
    console.warn(`Slow component detected: ${id} took ${actualDuration}ms`);
  }
}

function App() {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <Header />
      <Main />
      <Footer />
    </Profiler>
  );
}

// React DevTools Profiler
// 1. Install React DevTools browser extension
// 2. Open DevTools -> Profiler tab
// 3. Click record and interact with your app
// 4. Stop recording to see flame graph and ranked chart
```

## React.memo and PureComponent

### React.memo for Functional Components
```javascript
import React, { memo, useState } from 'react';

// Without memo - re-renders on every parent update
function ExpensiveChild({ name, age }) {
  console.log('ExpensiveChild rendered');
  
  // Simulate expensive computation
  const expensiveCalculation = () => {
    let result = 0;
    for (let i = 0; i < 1000000; i++) {
      result += i;
    }
    return result;
  };

  return (
    <div>
      <h3>{name}</h3>
      <p>Age: {age}</p>
      <p>Calculation: {expensiveCalculation()}</p>
    </div>
  );
}

// With memo - only re-renders when props change
const OptimizedChild = memo(function OptimizedChild({ name, age }) {
  console.log('OptimizedChild rendered');
  
  const expensiveCalculation = () => {
    let result = 0;
    for (let i = 0; i < 1000000; i++) {
      result += i;
    }
    return result;
  };

  return (
    <div>
      <h3>{name}</h3>
      <p>Age: {age}</p>
      <p>Calculation: {expensiveCalculation()}</p>
    </div>
  );
});

// Custom comparison function
const ChildWithCustomComparison = memo(
  function ChildWithCustomComparison({ user, settings }) {
    return (
      <div>
        <h3>{user.name}</h3>
        <p>Theme: {settings.theme}</p>
      </div>
    );
  },
  (prevProps, nextProps) => {
    // Only re-render if user name or theme changes
    return (
      prevProps.user.name === nextProps.user.name &&
      prevProps.settings.theme === nextProps.settings.theme
    );
  }
);

function ParentComponent() {
  const [count, setCount] = useState(0);
  const [user] = useState({ name: 'John', age: 30 });

  return (
    <div>
      <button onClick={() => setCount(count + 1)}>
        Count: {count}
      </button>
      
      {/* This will re-render on every count change */}
      <ExpensiveChild name={user.name} age={user.age} />
      
      {/* This will NOT re-render when count changes */}
      <OptimizedChild name={user.name} age={user.age} />
    </div>
  );
}
```

### PureComponent for Class Components
```javascript
import React, { Component, PureComponent } from 'react';

// Regular Component - always re-renders
class RegularComponent extends Component {
  render() {
    console.log('RegularComponent rendered');
    return <div>{this.props.name}</div>;
  }
}

// PureComponent - shallow comparison of props and state
class OptimizedComponent extends PureComponent {
  render() {
    console.log('OptimizedComponent rendered');
    return <div>{this.props.name}</div>;
  }
}

// Manual shouldComponentUpdate
class ManualOptimizedComponent extends Component {
  shouldComponentUpdate(nextProps, nextState) {
    // Custom logic for when to re-render
    return (
      this.props.name !== nextProps.name ||
      this.props.age !== nextProps.age
    );
  }

  render() {
    console.log('ManualOptimizedComponent rendered');
    return (
      <div>
        {this.props.name} - {this.props.age}
      </div>
    );
  }
}
```

## useMemo and useCallback

### useMemo for Expensive Calculations
```javascript
import React, { useState, useMemo } from 'react';

function ExpensiveComponent() {
  const [count, setCount] = useState(0);
  const [items, setItems] = useState([]);
  const [filter, setFilter] = useState('');

  // Without useMemo - recalculates on every render
  const expensiveValue = items
    .filter(item => item.name.includes(filter))
    .map(item => ({
      ...item,
      processed: true,
      timestamp: Date.now()
    }))
    .sort((a, b) => a.name.localeCompare(b.name));

  // With useMemo - only recalculates when dependencies change
  const optimizedValue = useMemo(() => {
    console.log('Expensive calculation running...');
    
    return items
      .filter(item => item.name.includes(filter))
      .map(item => ({
        ...item,
        processed: true,
        timestamp: Date.now()
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [items, filter]); // Only recalculate when items or filter changes

  // Complex object creation
  const userStats = useMemo(() => {
    const stats = {
      total: items.length,
      filtered: optimizedValue.length,
      categories: {},
    };

    items.forEach(item => {
      if (!stats.categories[item.category]) {
        stats.categories[item.category] = 0;
      }
      stats.categories[item.category]++;
    });

    return stats;
  }, [items, optimizedValue]);

  return (
    <div>
      <button onClick={() => setCount(count + 1)}>
        Count: {count} {/* This won't trigger expensive calculation */}
      </button>
      
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter items..."
      />
      
      <div>Total Items: {userStats.total}</div>
      <div>Filtered Items: {userStats.filtered}</div>
      
      <ul>
        {optimizedValue.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

### useCallback for Function Memoization
```javascript
import React, { useState, useCallback, memo } from 'react';

// Child component that receives a function prop
const ChildComponent = memo(function ChildComponent({ onItemClick, items }) {
  console.log('ChildComponent rendered');
  
  return (
    <ul>
      {items.map(item => (
        <li key={item.id} onClick={() => onItemClick(item)}>
          {item.name}
        </li>
      ))}
    </ul>
  );
});

function ParentComponent() {
  const [count, setCount] = useState(0);
  const [items, setItems] = useState([
    { id: 1, name: 'Item 1' },
    { id: 2, name: 'Item 2' },
    { id: 3, name: 'Item 3' },
  ]);

  // Without useCallback - new function on every render
  const handleItemClick = (item) => {
    console.log('Clicked:', item.name);
    // Some logic here
  };

  // With useCallback - same function reference unless dependencies change
  const memoizedHandleItemClick = useCallback((item) => {
    console.log('Clicked:', item.name);
    // Same logic, but memoized
  }, []); // Empty dependency array means function never changes

  // Callback with dependencies
  const handleItemClickWithDep = useCallback((item) => {
    console.log(`Clicked ${item.name} (count: ${count})`);
  }, [count]); // Function updates when count changes

  // Complex callback with multiple dependencies
  const handleComplexAction = useCallback((item, action) => {
    const timestamp = Date.now();
    console.log(`${action} on ${item.name} at ${timestamp}`);
    
    // Update items based on action
    setItems(prevItems => 
      prevItems.map(i => 
        i.id === item.id 
          ? { ...i, lastAction: action, timestamp }
          : i
      )
    );
  }, [setItems]); // setItems is stable, so this is safe

  return (
    <div>
      <button onClick={() => setCount(count + 1)}>
        Count: {count}
      </button>
      
      {/* This will cause ChildComponent to re-render every time */}
      <ChildComponent 
        items={items} 
        onItemClick={handleItemClick} 
      />
      
      {/* This will NOT cause unnecessary re-renders */}
      <ChildComponent 
        items={items} 
        onItemClick={memoizedHandleItemClick} 
      />
    </div>
  );
}
```

### When NOT to use useMemo/useCallback
```javascript
import React, { useState, useMemo, useCallback } from 'react';

function OverOptimizedComponent() {
  const [count, setCount] = useState(0);

  // ❌ DON'T: Memoizing primitive values
  const doubledCount = useMemo(() => count * 2, [count]);
  
  // ✅ DO: Just calculate directly
  const directDoubledCount = count * 2;

  // ❌ DON'T: Memoizing simple objects that change often
  const userInfo = useMemo(() => ({
    name: 'John',
    count: count
  }), [count]);

  // ❌ DON'T: Memoizing functions that recreate on every dependency change
  const expensiveCallback = useCallback(() => {
    return count * Math.random();
  }, [count]); // Changes every time count changes anyway

  // ✅ DO: Memoize when you have expensive calculations
  const expensiveCalculation = useMemo(() => {
    console.log('Expensive calculation...');
    let result = 0;
    for (let i = 0; i < 10000000; i++) {
      result += Math.sqrt(i);
    }
    return result;
  }, []); // No dependencies = calculate once

  return (
    <div>
      <p>Count: {count}</p>
      <p>Doubled: {directDoubledCount}</p>
      <p>Expensive: {expensiveCalculation}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

## Code Splitting and Lazy Loading

### React.lazy and Suspense
```javascript
import React, { Suspense, lazy, useState } from 'react';

// Lazy load components
const HeavyComponent = lazy(() => import('./HeavyComponent'));
const Dashboard = lazy(() => import('./Dashboard'));
const Settings = lazy(() => import('./Settings'));

// Dynamic imports with error handling
const LazyComponentWithErrorHandling = lazy(() => 
  import('./SomeComponent').catch(() => {
    // Fallback to a default component if import fails
    return { default: () => <div>Failed to load component</div> };
  })
);

function App() {
  const [currentView, setCurrentView] = useState('home');

  return (
    <div>
      <nav>
        <button onClick={() => setCurrentView('home')}>Home</button>
        <button onClick={() => setCurrentView('dashboard')}>Dashboard</button>
        <button onClick={() => setCurrentView('settings')}>Settings</button>
      </nav>

      <Suspense fallback={<LoadingSpinner />}>
        {currentView === 'home' && <HomeComponent />}
        {currentView === 'dashboard' && <Dashboard />}
        {currentView === 'settings' && <Settings />}
      </Suspense>

      {/* Load heavy component only when needed */}
      <Suspense fallback={<div>Loading heavy component...</div>}>
        <HeavyComponent />
      </Suspense>
    </div>
  );
}

function LoadingSpinner() {
  return (
    <div className="loading-spinner">
      <div className="spinner"></div>
      <p>Loading...</p>
    </div>
  );
}

// Route-based code splitting with React Router
import { BrowserRouter, Routes, Route } from 'react-router-dom';

const Home = lazy(() => import('./pages/Home'));
const About = lazy(() => import('./pages/About'));
const Contact = lazy(() => import('./pages/Contact'));

function AppWithRouter() {
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
```

### Dynamic Imports
```javascript
import React, { useState } from 'react';

function DynamicImportExample() {
  const [module, setModule] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadModule = async (moduleName) => {
    setLoading(true);
    try {
      let importedModule;
      
      switch (moduleName) {
        case 'chart':
          importedModule = await import('./ChartComponent');
          break;
        case 'editor':
          importedModule = await import('./TextEditor');
          break;
        case 'calendar':
          importedModule = await import('./Calendar');
          break;
        default:
          throw new Error('Unknown module');
      }
      
      setModule(importedModule.default);
    } catch (error) {
      console.error('Failed to load module:', error);
    } finally {
      setLoading(false);
    }
  };

  // Load external libraries dynamically
  const loadExternalLibrary = async () => {
    try {
      // Dynamic import of a large library
      const { default: moment } = await import('moment');
      console.log('Current time:', moment().format());
      
      // Load multiple modules
      const [{ default: lodash }, { default: axios }] = await Promise.all([
        import('lodash'),
        import('axios')
      ]);
      
      console.log('Libraries loaded');
    } catch (error) {
      console.error('Failed to load libraries:', error);
    }
  };

  return (
    <div>
      <h2>Dynamic Import Example</h2>
      
      <div>
        <button onClick={() => loadModule('chart')}>
          Load Chart Component
        </button>
        <button onClick={() => loadModule('editor')}>
          Load Text Editor
        </button>
        <button onClick={() => loadModule('calendar')}>
          Load Calendar
        </button>
        <button onClick={loadExternalLibrary}>
          Load External Libraries
        </button>
      </div>

      {loading && <p>Loading module...</p>}
      
      {module && React.createElement(module)}
    </div>
  );
}
```

### Preloading Components
```javascript
import React, { useEffect } from 'react';

// Preload components that will likely be needed
const preloadComponent = (importFunc) => {
  const componentImport = importFunc();
  return componentImport;
};

function ComponentPreloader() {
  useEffect(() => {
    // Preload components on app start
    const timer = setTimeout(() => {
      preloadComponent(() => import('./Dashboard'));
      preloadComponent(() => import('./Settings'));
    }, 2000); // Preload after 2 seconds

    return () => clearTimeout(timer);
  }, []);

  // Preload on user interaction
  const handleMouseEnter = () => {
    preloadComponent(() => import('./HeavyComponent'));
  };

  return (
    <div>
      <button onMouseEnter={handleMouseEnter}>
        Hover to preload heavy component
      </button>
    </div>
  );
}

// Resource hints for preloading
// Add to your HTML head:
// <link rel="prefetch" href="/chunk-dashboard.js">
// <link rel="preload" href="/chunk-critical.js" as="script">
```

## Virtual Scrolling

### Basic Virtual List Implementation
```javascript
import React, { useState, useMemo, useRef, useEffect } from 'react';

function VirtualList({ items, itemHeight = 50, containerHeight = 400 }) {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef();

  const totalHeight = items.length * itemHeight;
  const visibleStart = Math.floor(scrollTop / itemHeight);
  const visibleEnd = Math.min(
    visibleStart + Math.ceil(containerHeight / itemHeight) + 1,
    items.length
  );

  const visibleItems = useMemo(() => {
    return items.slice(visibleStart, visibleEnd).map((item, index) => ({
      ...item,
      index: visibleStart + index,
    }));
  }, [items, visibleStart, visibleEnd]);

  const offsetY = visibleStart * itemHeight;

  const handleScroll = (e) => {
    setScrollTop(e.currentTarget.scrollTop);
  };

  return (
    <div
      ref={containerRef}
      style={{
        height: containerHeight,
        overflow: 'auto',
      }}
      onScroll={handleScroll}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div style={{ transform: `translateY(${offsetY}px)` }}>
          {visibleItems.map((item) => (
            <div
              key={item.id}
              style={{
                height: itemHeight,
                padding: '10px',
                borderBottom: '1px solid #eee',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <span>#{item.index}: {item.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Usage with large dataset
function VirtualListExample() {
  const items = useMemo(() => 
    Array.from({ length: 10000 }, (_, i) => ({
      id: i,
      name: `Item ${i + 1}`,
      description: `Description for item ${i + 1}`,
    }))
  , []);

  return (
    <div>
      <h2>Virtual List (10,000 items)</h2>
      <VirtualList 
        items={items} 
        itemHeight={60} 
        containerHeight={500} 
      />
    </div>
  );
}
```

### Advanced Virtual List with Variable Heights
```javascript
import React, { useState, useRef, useEffect, useCallback } from 'react';

function VariableVirtualList({ items, estimatedItemHeight = 50, containerHeight = 400 }) {
  const [scrollTop, setScrollTop] = useState(0);
  const [itemHeights, setItemHeights] = useState(new Map());
  const containerRef = useRef();
  const itemRefs = useRef(new Map());

  // Calculate positions based on measured heights
  const { totalHeight, itemPositions } = useMemo(() => {
    let totalHeight = 0;
    const positions = new Map();

    items.forEach((item, index) => {
      positions.set(index, totalHeight);
      const height = itemHeights.get(index) || estimatedItemHeight;
      totalHeight += height;
    });

    return { totalHeight, itemPositions: positions };
  }, [items, itemHeights, estimatedItemHeight]);

  // Find visible range
  const visibleRange = useMemo(() => {
    let start = 0;
    let end = items.length;

    // Binary search for start
    for (let i = 0; i < items.length; i++) {
      if ((itemPositions.get(i) || 0) >= scrollTop) {
        start = Math.max(0, i - 1);
        break;
      }
    }

    // Find end
    for (let i = start; i < items.length; i++) {
      if ((itemPositions.get(i) || 0) > scrollTop + containerHeight) {
        end = i + 1;
        break;
      }
    }

    return { start, end };
  }, [scrollTop, containerHeight, itemPositions, items.length]);

  // Measure item heights
  const measureItem = useCallback((index, element) => {
    if (element) {
      const height = element.getBoundingClientRect().height;
      setItemHeights(prev => {
        const newHeights = new Map(prev);
        if (newHeights.get(index) !== height) {
          newHeights.set(index, height);
          return newHeights;
        }
        return prev;
      });
    }
  }, []);

  const handleScroll = (e) => {
    setScrollTop(e.currentTarget.scrollTop);
  };

  return (
    <div
      ref={containerRef}
      style={{
        height: containerHeight,
        overflow: 'auto',
      }}
      onScroll={handleScroll}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        {items.slice(visibleRange.start, visibleRange.end).map((item, index) => {
          const actualIndex = visibleRange.start + index;
          const top = itemPositions.get(actualIndex) || 0;

          return (
            <div
              key={item.id}
              ref={(el) => measureItem(actualIndex, el)}
              style={{
                position: 'absolute',
                top,
                left: 0,
                right: 0,
                padding: '10px',
                borderBottom: '1px solid #eee',
              }}
            >
              <h3>{item.title}</h3>
              <p>{item.content}</p>
              <small>Item #{actualIndex}</small>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

## Bundle Optimization

### Webpack Bundle Analyzer
```javascript
// Install webpack-bundle-analyzer
// npm install --save-dev webpack-bundle-analyzer

// For Create React App (need to eject or use CRACO)
// package.json scripts:
{
  "analyze": "npm run build && npx webpack-bundle-analyzer build/static/js/*.js"
}

// For Vite
// vite.config.js
import { defineConfig } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    visualizer({
      filename: 'dist/stats.html',
      open: true
    })
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          ui: ['@mui/material', '@emotion/react'],
        }
      }
    }
  }
});
```

### Tree Shaking Optimization
```javascript
// ❌ Don't import entire libraries
import * as _ from 'lodash';
import { Button, TextField, Dialog } from '@mui/material';

// ✅ Do import only what you need
import debounce from 'lodash/debounce';
import get from 'lodash/get';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';

// Configure babel for better tree shaking
// babel.config.js
module.exports = {
  plugins: [
    ['import', {
      libraryName: '@mui/material',
      libraryDirectory: '',
      camel2DashComponentName: false
    }]
  ]
};

// Mark packages as side-effect free
// package.json
{
  "sideEffects": false
}

// Or specify which files have side effects
{
  "sideEffects": ["*.css", "*.scss", "./src/polyfills.js"]
}
```

### Image Optimization
```javascript
import React from 'react';

// Use appropriate image formats
function OptimizedImages() {
  return (
    <div>
      {/* Use WebP with fallback */}
      <picture>
        <source srcSet="image.webp" type="image/webp" />
        <source srcSet="image.jpg" type="image/jpeg" />
        <img src="image.jpg" alt="Description" />
      </picture>

      {/* Responsive images */}
      <img
        srcSet="
          image-320.jpg 320w,
          image-640.jpg 640w,
          image-1280.jpg 1280w
        "
        sizes="(max-width: 320px) 280px, (max-width: 640px) 600px, 1200px"
        src="image-640.jpg"
        alt="Responsive image"
      />

      {/* Lazy loading */}
      <img
        src="image.jpg"
        alt="Lazy loaded"
        loading="lazy"
        decoding="async"
      />
    </div>
  );
}

// Image optimization with next/image (if using Next.js)
import Image from 'next/image';

function NextImageExample() {
  return (
    <Image
      src="/image.jpg"
      alt="Optimized image"
      width={500}
      height={300}
      placeholder="blur"
      blurDataURL="data:image/jpeg;base64,..."
    />
  );
}
```

## Profiling and Debugging

### React DevTools Profiler
```javascript
import React, { Profiler, useState } from 'react';

function onRenderCallback(id, phase, actualDuration, baseDuration, startTime, commitTime, interactions) {
  // Log performance data
  console.group(`Profiler: ${id}`);
  console.log('Phase:', phase);
  console.log('Actual duration:', actualDuration);
  console.log('Base duration:', baseDuration);
  console.log('Start time:', startTime);
  console.log('Commit time:', commitTime);
  console.log('Interactions:', interactions);
  console.groupEnd();

  // Send to analytics
  if (actualDuration > 16) { // More than one frame
    analytics.track('slow-component', {
      componentId: id,
      duration: actualDuration,
      phase
    });
  }
}

function ProfiledApp() {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <Header />
      <Profiler id="MainContent" onRender={onRenderCallback}>
        <MainContent />
      </Profiler>
      <Footer />
    </Profiler>
  );
}
```

### Performance Monitoring
```javascript
import React, { useEffect } from 'react';

// Custom hook for performance monitoring
function usePerformanceMonitor(componentName) {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      console.log(`${componentName} was active for ${duration}ms`);
      
      // Track long-running components
      if (duration > 100) {
        console.warn(`Long-running component: ${componentName}`);
      }
    };
  }, [componentName]);
}

// Web Vitals monitoring
function useWebVitals() {
  useEffect(() => {
    import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
      getCLS(console.log);
      getFID(console.log);
      getFCP(console.log);
      getLCP(console.log);
      getTTFB(console.log);
    });
  }, []);
}

function MonitoredComponent() {
  usePerformanceMonitor('MonitoredComponent');
  useWebVitals();

  return <div>Component with performance monitoring</div>;
}
```

## Performance Patterns

### Compound Components Pattern
```javascript
import React, { createContext, useContext, useState } from 'react';

// Context for compound component
const AccordionContext = createContext();

function Accordion({ children, defaultOpen = null }) {
  const [openItem, setOpenItem] = useState(defaultOpen);

  const toggle = (id) => {
    setOpenItem(openItem === id ? null : id);
  };

  return (
    <AccordionContext.Provider value={{ openItem, toggle }}>
      <div className="accordion">{children}</div>
    </AccordionContext.Provider>
  );
}

function AccordionItem({ id, children }) {
  return (
    <div className="accordion-item">
      {children}
    </div>
  );
}

function AccordionHeader({ id, children }) {
  const { openItem, toggle } = useContext(AccordionContext);
  
  return (
    <button
      className="accordion-header"
      onClick={() => toggle(id)}
      aria-expanded={openItem === id}
    >
      {children}
    </button>
  );
}

function AccordionPanel({ id, children }) {
  const { openItem } = useContext(AccordionContext);
  
  if (openItem !== id) return null;
  
  return (
    <div className="accordion-panel">
      {children}
    </div>
  );
}

// Usage
function AccordionExample() {
  return (
    <Accordion defaultOpen="item1">
      <AccordionItem id="item1">
        <AccordionHeader id="item1">Header 1</AccordionHeader>
        <AccordionPanel id="item1">Panel 1 content</AccordionPanel>
      </AccordionItem>
      
      <AccordionItem id="item2">
        <AccordionHeader id="item2">Header 2</AccordionHeader>
        <AccordionPanel id="item2">Panel 2 content</AccordionPanel>
      </AccordionItem>
    </Accordion>
  );
}
```

### Render Props Pattern for Performance
```javascript
import React, { useState, useCallback } from 'react';

function DataProvider({ render }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/data');
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }, []);

  return render({ data, loading, fetchData });
}

// Optimized usage
function OptimizedDataConsumer() {
  return (
    <DataProvider
      render={({ data, loading, fetchData }) => (
        <div>
          {loading ? (
            <LoadingSpinner />
          ) : (
            <DataList data={data} onRefresh={fetchData} />
          )}
        </div>
      )}
    />
  );
}
```

### Intersection Observer for Lazy Loading
```javascript
import React, { useEffect, useRef, useState } from 'react';

function useIntersectionObserver(options = {}) {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const targetRef = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setIsIntersecting(entry.isIntersecting),
      options
    );

    if (targetRef.current) {
      observer.observe(targetRef.current);
    }

    return () => observer.disconnect();
  }, [options]);

  return [targetRef, isIntersecting];
}

function LazyImage({ src, alt, ...props }) {
  const [imgRef, isVisible] = useIntersectionObserver({
    threshold: 0.1,
    rootMargin: '50px'
  });
  const [isLoaded, setIsLoaded] = useState(false);

  return (
    <div ref={imgRef} {...props}>
      {isVisible && (
        <img
          src={src}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          style={{
            opacity: isLoaded ? 1 : 0,
            transition: 'opacity 0.3s'
          }}
        />
      )}
    </div>
  );
}

function LazyComponent({ children }) {
  const [componentRef, isVisible] = useIntersectionObserver({
    threshold: 0.1
  });

  return (
    <div ref={componentRef}>
      {isVisible ? children : <div>Loading...</div>}
    </div>
  );
}
```

---

*Continue to: [07-react-testing.md](./07-react-testing.md)*
