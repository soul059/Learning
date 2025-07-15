# React Hooks

## Table of Contents
- [Introduction to Hooks](#introduction-to-hooks)
- [Built-in Hooks](#built-in-hooks)
- [Custom Hooks](#custom-hooks)
- [Hook Rules and Best Practices](#hook-rules-and-best-practices)
- [Advanced Hook Patterns](#advanced-hook-patterns)
- [Performance Hooks](#performance-hooks)
- [Common Hook Recipes](#common-hook-recipes)

## Introduction to Hooks

Hooks are functions that let you "hook into" React features. They allow you to use state and other React features in function components.

### Why Hooks?
- **Simpler code**: No need for class components
- **Better reusability**: Logic can be extracted into custom hooks
- **Easier testing**: Pure functions are easier to test
- **Better performance**: Optimizations like `useMemo` and `useCallback`

### Hook Rules
1. **Only call hooks at the top level** - Never inside loops, conditions, or nested functions
2. **Only call hooks from React functions** - React function components or custom hooks

```javascript
// ❌ Wrong - conditional hook
function MyComponent({ condition }) {
  if (condition) {
    const [state, setState] = useState(0); // Don't do this!
  }
}

// ✅ Correct - hook at top level
function MyComponent({ condition }) {
  const [state, setState] = useState(0);
  
  if (condition) {
    // Use the state here
  }
}
```

## Built-in Hooks

### useState
Manages local component state.

```javascript
import { useState } from 'react';

function Counter() {
  // Basic usage
  const [count, setCount] = useState(0);
  
  // Multiple states
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  
  // Object state
  const [user, setUser] = useState({
    name: '',
    email: '',
    age: 0
  });
  
  // Array state
  const [items, setItems] = useState([]);
  
  // Lazy initial state (for expensive computations)
  const [data, setData] = useState(() => {
    const saved = localStorage.getItem('data');
    return saved ? JSON.parse(saved) : [];
  });

  // Functional updates
  const increment = () => setCount(prev => prev + 1);
  const decrement = () => setCount(prev => prev - 1);
  
  // Object updates
  const updateUser = (field, value) => {
    setUser(prev => ({ ...prev, [field]: value }));
  };
  
  // Array updates
  const addItem = (item) => {
    setItems(prev => [...prev, item]);
  };
  
  const removeItem = (id) => {
    setItems(prev => prev.filter(item => item.id !== id));
  };

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>+</button>
      <button onClick={decrement}>-</button>
      
      <input 
        value={user.name}
        onChange={(e) => updateUser('name', e.target.value)}
        placeholder="Name"
      />
    </div>
  );
}
```

### useEffect
Handles side effects in function components.

```javascript
import { useState, useEffect } from 'react';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Effect with dependency array
  useEffect(() => {
    if (!userId) return;
    
    setLoading(true);
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(userData => {
        setUser(userData);
        setLoading(false);
      })
      .catch(error => {
        console.error('Failed to fetch user:', error);
        setLoading(false);
      });
  }, [userId]); // Runs when userId changes

  // Effect with cleanup
  useEffect(() => {
    const timer = setInterval(() => {
      console.log('Timer tick');
    }, 1000);

    // Cleanup function
    return () => {
      clearInterval(timer);
    };
  }, []); // Empty array = run once on mount

  // Effect without dependencies (runs on every render)
  useEffect(() => {
    document.title = user ? `Profile: ${user.name}` : 'Loading...';
  });

  // Multiple effects for separation of concerns
  useEffect(() => {
    // Track page view
    analytics.track('page_view', { page: 'user_profile', userId });
  }, [userId]);

  useEffect(() => {
    // Set up keyboard shortcuts
    const handleKeyPress = (e) => {
      if (e.key === 'Escape') {
        // Handle escape
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, []);

  if (loading) return <div>Loading...</div>;
  if (!user) return <div>User not found</div>;

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}
```

### useContext
Consumes context values without nesting.

```javascript
import { createContext, useContext, useState } from 'react';

// Create context
const ThemeContext = createContext();
const UserContext = createContext();

// Provider components
function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

function UserProvider({ children }) {
  const [user, setUser] = useState(null);
  
  const login = (userData) => setUser(userData);
  const logout = () => setUser(null);

  return (
    <UserContext.Provider value={{ user, login, logout }}>
      {children}
    </UserContext.Provider>
  );
}

// Custom hooks for context
function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

function useUser() {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}

// Component using context
function Header() {
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useUser();

  return (
    <header className={`header header-${theme}`}>
      <h1>My App</h1>
      <button onClick={toggleTheme}>
        Switch to {theme === 'light' ? 'dark' : 'light'} mode
      </button>
      {user ? (
        <div>
          Welcome, {user.name}!
          <button onClick={logout}>Logout</button>
        </div>
      ) : (
        <button>Login</button>
      )}
    </header>
  );
}

// App with providers
function App() {
  return (
    <ThemeProvider>
      <UserProvider>
        <Header />
        <main>
          {/* Rest of app */}
        </main>
      </UserProvider>
    </ThemeProvider>
  );
}
```

### useReducer
Manages complex state logic with a reducer function.

```javascript
import { useReducer } from 'react';

// Reducer function
function todoReducer(state, action) {
  switch (action.type) {
    case 'ADD_TODO':
      return [...state, {
        id: Date.now(),
        text: action.text,
        completed: false
      }];
    
    case 'TOGGLE_TODO':
      return state.map(todo =>
        todo.id === action.id 
          ? { ...todo, completed: !todo.completed }
          : todo
      );
    
    case 'DELETE_TODO':
      return state.filter(todo => todo.id !== action.id);
    
    case 'EDIT_TODO':
      return state.map(todo =>
        todo.id === action.id 
          ? { ...todo, text: action.text }
          : todo
      );
    
    case 'CLEAR_COMPLETED':
      return state.filter(todo => !todo.completed);
    
    default:
      throw new Error(`Unknown action type: ${action.type}`);
  }
}

function TodoApp() {
  const [todos, dispatch] = useReducer(todoReducer, []);
  const [inputText, setInputText] = useState('');

  const addTodo = () => {
    if (inputText.trim()) {
      dispatch({ type: 'ADD_TODO', text: inputText });
      setInputText('');
    }
  };

  const completedCount = todos.filter(todo => todo.completed).length;
  const activeCount = todos.length - completedCount;

  return (
    <div>
      <h1>Todo App</h1>
      
      <div>
        <input 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && addTodo()}
          placeholder="Add a todo..."
        />
        <button onClick={addTodo}>Add</button>
      </div>

      <div>
        Active: {activeCount} | Completed: {completedCount}
        <button onClick={() => dispatch({ type: 'CLEAR_COMPLETED' })}>
          Clear Completed
        </button>
      </div>

      <ul>
        {todos.map(todo => (
          <li key={todo.id} className={todo.completed ? 'completed' : ''}>
            <input 
              type="checkbox"
              checked={todo.completed}
              onChange={() => dispatch({ type: 'TOGGLE_TODO', id: todo.id })}
            />
            <span>{todo.text}</span>
            <button onClick={() => dispatch({ type: 'DELETE_TODO', id: todo.id })}>
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Complex state with useReducer
const initialState = {
  user: null,
  posts: [],
  loading: false,
  error: null
};

function appReducer(state, action) {
  switch (action.type) {
    case 'FETCH_START':
      return { ...state, loading: true, error: null };
    
    case 'FETCH_SUCCESS':
      return { 
        ...state, 
        loading: false, 
        [action.dataType]: action.data 
      };
    
    case 'FETCH_ERROR':
      return { 
        ...state, 
        loading: false, 
        error: action.error 
      };
    
    case 'SET_USER':
      return { ...state, user: action.user };
    
    default:
      return state;
  }
}

function App() {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const fetchData = async (url, dataType) => {
    dispatch({ type: 'FETCH_START' });
    try {
      const response = await fetch(url);
      const data = await response.json();
      dispatch({ type: 'FETCH_SUCCESS', data, dataType });
    } catch (error) {
      dispatch({ type: 'FETCH_ERROR', error: error.message });
    }
  };

  return (
    <div>
      {state.loading && <div>Loading...</div>}
      {state.error && <div>Error: {state.error}</div>}
      {/* Render UI based on state */}
    </div>
  );
}
```

### useCallback
Memoizes callback functions to prevent unnecessary re-renders.

```javascript
import { useState, useCallback, memo } from 'react';

// Child component that should only re-render when necessary
const TodoItem = memo(({ todo, onToggle, onDelete }) => {
  console.log('TodoItem rendered:', todo.id);
  
  return (
    <li>
      <input 
        type="checkbox"
        checked={todo.completed}
        onChange={() => onToggle(todo.id)}
      />
      <span>{todo.text}</span>
      <button onClick={() => onDelete(todo.id)}>Delete</button>
    </li>
  );
});

function TodoList() {
  const [todos, setTodos] = useState([]);
  const [filter, setFilter] = useState('all');

  // Without useCallback, these functions are recreated on every render
  // causing TodoItem to re-render even when todo data hasn't changed
  
  // ❌ Without useCallback - causes unnecessary re-renders
  // const handleToggle = (id) => {
  //   setTodos(prev => prev.map(todo =>
  //     todo.id === id ? { ...todo, completed: !todo.completed } : todo
  //   ));
  // };

  // ✅ With useCallback - function reference stays same
  const handleToggle = useCallback((id) => {
    setTodos(prev => prev.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  }, []); // No dependencies, function never changes

  const handleDelete = useCallback((id) => {
    setTodos(prev => prev.filter(todo => todo.id !== id));
  }, []);

  // Callback with dependencies
  const handleFilteredDelete = useCallback((id) => {
    setTodos(prev => prev.filter(todo => todo.id !== id));
    // If filter changes, we might want to do something different
    if (filter === 'completed') {
      // Some filter-specific logic
    }
  }, [filter]); // Recreate when filter changes

  const filteredTodos = todos.filter(todo => {
    if (filter === 'active') return !todo.completed;
    if (filter === 'completed') return todo.completed;
    return true;
  });

  return (
    <div>
      <div>
        <button onClick={() => setFilter('all')}>All</button>
        <button onClick={() => setFilter('active')}>Active</button>
        <button onClick={() => setFilter('completed')}>Completed</button>
      </div>
      
      <ul>
        {filteredTodos.map(todo => (
          <TodoItem 
            key={todo.id}
            todo={todo}
            onToggle={handleToggle}
            onDelete={handleDelete}
          />
        ))}
      </ul>
    </div>
  );
}
```

### useMemo
Memoizes expensive calculations.

```javascript
import { useState, useMemo } from 'react';

function ExpensiveComponent({ items, filter, sortBy }) {
  const [searchTerm, setSearchTerm] = useState('');

  // Expensive calculation without useMemo
  // This runs on every render, even when items/filter/sortBy haven't changed
  // const expensiveValue = items
  //   .filter(item => item.category === filter)
  //   .sort((a, b) => a[sortBy] - b[sortBy])
  //   .map(item => ({ ...item, processed: true }));

  // ✅ With useMemo - only recalculates when dependencies change
  const expensiveValue = useMemo(() => {
    console.log('Calculating expensive value...');
    return items
      .filter(item => item.category === filter)
      .sort((a, b) => a[sortBy] - b[sortBy])
      .map(item => ({ ...item, processed: true }));
  }, [items, filter, sortBy]);

  // Filtered results based on search
  const searchResults = useMemo(() => {
    if (!searchTerm) return expensiveValue;
    
    return expensiveValue.filter(item =>
      item.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [expensiveValue, searchTerm]);

  // Complex object that should be memoized
  const chartData = useMemo(() => {
    return {
      labels: searchResults.map(item => item.name),
      datasets: [{
        data: searchResults.map(item => item.value),
        backgroundColor: searchResults.map(item => item.color)
      }]
    };
  }, [searchResults]);

  // Memoize derived state
  const statistics = useMemo(() => {
    const total = searchResults.reduce((sum, item) => sum + item.value, 0);
    const average = total / searchResults.length || 0;
    const max = Math.max(...searchResults.map(item => item.value));
    const min = Math.min(...searchResults.map(item => item.value));
    
    return { total, average, max, min };
  }, [searchResults]);

  return (
    <div>
      <input 
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search..."
      />
      
      <div>
        <p>Total: {statistics.total}</p>
        <p>Average: {statistics.average.toFixed(2)}</p>
        <p>Max: {statistics.max}</p>
        <p>Min: {statistics.min}</p>
      </div>

      <ul>
        {searchResults.map(item => (
          <li key={item.id}>{item.name}: {item.value}</li>
        ))}
      </ul>
    </div>
  );
}
```

### useRef
Creates a mutable reference that persists across renders.

```javascript
import { useRef, useEffect, useState } from 'react';

function RefExamples() {
  // DOM element references
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const scrollContainerRef = useRef(null);

  // Mutable value that doesn't trigger re-renders
  const renderCount = useRef(0);
  const previousValue = useRef();

  // Timer/interval references
  const timerRef = useRef(null);

  const [count, setCount] = useState(0);
  const [value, setValue] = useState('');

  // Track render count
  renderCount.current += 1;

  // Store previous value
  useEffect(() => {
    previousValue.current = value;
  });

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Timer management
  const startTimer = () => {
    if (timerRef.current) return; // Already running
    
    timerRef.current = setInterval(() => {
      setCount(prev => prev + 1);
    }, 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Imperative actions
  const focusInput = () => {
    inputRef.current?.focus();
  };

  const selectFile = () => {
    fileInputRef.current?.click();
  };

  const scrollToTop = () => {
    scrollContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollToBottom = () => {
    const container = scrollContainerRef.current;
    if (container) {
      container.scrollTo({ 
        top: container.scrollHeight, 
        behavior: 'smooth' 
      });
    }
  };

  return (
    <div>
      <h2>useRef Examples</h2>
      
      <div>
        <p>Render count: {renderCount.current}</p>
        <p>Previous value: {previousValue.current}</p>
        <p>Current value: {value}</p>
        <p>Timer count: {count}</p>
      </div>

      <div>
        <input 
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Type something..."
        />
        <button onClick={focusInput}>Focus Input</button>
      </div>

      <div>
        <input 
          ref={fileInputRef}
          type="file"
          style={{ display: 'none' }}
          onChange={(e) => console.log('File selected:', e.target.files[0])}
        />
        <button onClick={selectFile}>Select File</button>
      </div>

      <div>
        <button onClick={startTimer}>Start Timer</button>
        <button onClick={stopTimer}>Stop Timer</button>
      </div>

      <div 
        ref={scrollContainerRef}
        style={{ 
          height: '200px', 
          overflow: 'auto', 
          border: '1px solid #ccc' 
        }}
      >
        {Array.from({ length: 100 }, (_, i) => (
          <div key={i} style={{ padding: '10px' }}>
            Item {i + 1}
          </div>
        ))}
      </div>

      <div>
        <button onClick={scrollToTop}>Scroll to Top</button>
        <button onClick={scrollToBottom}>Scroll to Bottom</button>
      </div>
    </div>
  );
}

// Custom hook using useRef for previous value
function usePrevious(value) {
  const ref = useRef();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}

// Usage of custom hook
function Counter() {
  const [count, setCount] = useState(0);
  const previousCount = usePrevious(count);

  return (
    <div>
      <p>Current: {count}</p>
      <p>Previous: {previousCount}</p>
      <button onClick={() => setCount(count + 1)}>+</button>
    </div>
  );
}
```

## Custom Hooks

Custom hooks are functions that use built-in hooks and can be shared between components.

### Basic Custom Hooks

```javascript
// useToggle - Simple boolean state toggle
function useToggle(initialValue = false) {
  const [value, setValue] = useState(initialValue);
  
  const toggle = useCallback(() => setValue(prev => !prev), []);
  const setTrue = useCallback(() => setValue(true), []);
  const setFalse = useCallback(() => setValue(false), []);
  
  return [value, { toggle, setTrue, setFalse }];
}

// Usage
function Modal() {
  const [isOpen, { toggle, setTrue, setFalse }] = useToggle(false);
  
  return (
    <div>
      <button onClick={setTrue}>Open Modal</button>
      {isOpen && (
        <div className="modal">
          <p>Modal content</p>
          <button onClick={setFalse}>Close</button>
        </div>
      )}
    </div>
  );
}

// useLocalStorage - Sync state with localStorage
function useLocalStorage(key, initialValue) {
  // Get from local storage then parse stored json or return initialValue
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error('Error reading localStorage key "' + key + '":', error);
      return initialValue;
    }
  });

  // Return a wrapped version of useState's setter function that persists the new value to localStorage
  const setValue = useCallback((value) => {
    try {
      // Allow value to be a function so we have the same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error('Error setting localStorage key "' + key + '":', error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue];
}

// Usage
function Settings() {
  const [theme, setTheme] = useLocalStorage('theme', 'light');
  const [language, setLanguage] = useLocalStorage('language', 'en');
  
  return (
    <div>
      <select value={theme} onChange={(e) => setTheme(e.target.value)}>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
      
      <select value={language} onChange={(e) => setLanguage(e.target.value)}>
        <option value="en">English</option>
        <option value="es">Spanish</option>
      </select>
    </div>
  );
}
```

### Advanced Custom Hooks

```javascript
// useFetch - Data fetching with loading and error states
function useFetch(url, options = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(url, options);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [url, JSON.stringify(options)]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const refetch = useCallback(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch };
}

// Usage
function UserProfile({ userId }) {
  const { data: user, loading, error, refetch } = useFetch(`/api/users/${userId}`);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!user) return <div>No user found</div>;

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      <button onClick={refetch}>Refresh</button>
    </div>
  );
}

// useDebounce - Debounce a value
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

// Usage
function SearchInput() {
  const [searchTerm, setSearchTerm] = useState('');
  const debouncedSearchTerm = useDebounce(searchTerm, 300);

  useEffect(() => {
    if (debouncedSearchTerm) {
      // Perform search
      console.log('Searching for:', debouncedSearchTerm);
    }
  }, [debouncedSearchTerm]);

  return (
    <input
      value={searchTerm}
      onChange={(e) => setSearchTerm(e.target.value)}
      placeholder="Search..."
    />
  );
}

// useWindowSize - Track window dimensions
function useWindowSize() {
  const [windowSize, setWindowSize] = useState({
    width: undefined,
    height: undefined,
  });

  useEffect(() => {
    function handleResize() {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }

    window.addEventListener('resize', handleResize);
    handleResize(); // Call handler right away so state gets updated with initial window size

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return windowSize;
}

// Usage
function ResponsiveComponent() {
  const { width, height } = useWindowSize();

  return (
    <div>
      <p>Window size: {width} x {height}</p>
      {width < 768 ? (
        <MobileLayout />
      ) : (
        <DesktopLayout />
      )}
    </div>
  );
}

// useInterval - Declarative interval
function useInterval(callback, delay) {
  const savedCallback = useRef();

  // Remember the latest callback.
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  // Set up the interval.
  useEffect(() => {
    function tick() {
      savedCallback.current();
    }
    if (delay !== null) {
      let id = setInterval(tick, delay);
      return () => clearInterval(id);
    }
  }, [delay]);
}

// Usage
function Timer() {
  const [count, setCount] = useState(0);
  const [isRunning, setIsRunning] = useState(true);

  useInterval(() => {
    setCount(count + 1);
  }, isRunning ? 1000 : null);

  return (
    <div>
      <h1>{count}</h1>
      <button onClick={() => setIsRunning(!isRunning)}>
        {isRunning ? 'Pause' : 'Start'}
      </button>
    </div>
  );
}
```

## Hook Rules and Best Practices

### Rules of Hooks

```javascript
// ❌ Don't call hooks inside loops, conditions, or nested functions
function BadComponent({ condition, items }) {
  if (condition) {
    const [state, setState] = useState(0); // Wrong!
  }

  for (let item of items) {
    const [itemState, setItemState] = useState(item); // Wrong!
  }

  function nestedFunction() {
    const [nested, setNested] = useState(0); // Wrong!
  }
}

// ✅ Always call hooks at the top level
function GoodComponent({ condition, items }) {
  const [state, setState] = useState(0);
  const [itemStates, setItemStates] = useState({});
  const [nested, setNested] = useState(0);

  // Use conditional logic inside the hook or in the render
  if (condition) {
    // Use state here
  }

  // Handle arrays of state differently
  const updateItemState = (id, value) => {
    setItemStates(prev => ({ ...prev, [id]: value }));
  };
}

// ❌ Don't call hooks from regular JavaScript functions
function regularFunction() {
  const [state, setState] = useState(0); // Wrong!
}

// ✅ Only call hooks from React function components or custom hooks
function useCustomHook() {
  const [state, setState] = useState(0); // Correct!
  return [state, setState];
}

function MyComponent() {
  const [state, setState] = useState(0); // Correct!
  const [customState, setCustomState] = useCustomHook(); // Correct!
}
```

### Best Practices

```javascript
// ✅ Use multiple state variables for unrelated data
function UserProfile() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [age, setAge] = useState(0);
  
  // Instead of:
  // const [user, setUser] = useState({ name: '', email: '', age: 0 });
}

// ✅ Group related state into objects
function FormComponent() {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: ''
  });
  
  const [formState, setFormState] = useState({
    isSubmitting: false,
    errors: {},
    touched: {}
  });
}

// ✅ Use useCallback for expensive functions passed to child components
function ParentComponent() {
  const [items, setItems] = useState([]);
  
  const handleItemClick = useCallback((id) => {
    // Expensive operation
    setItems(prev => prev.map(item => 
      item.id === id ? { ...item, clicked: true } : item
    ));
  }, []); // Dependencies array is empty, function never changes
  
  return (
    <div>
      {items.map(item => (
        <ExpensiveChildComponent 
          key={item.id}
          item={item}
          onClick={handleItemClick}
        />
      ))}
    </div>
  );
}

// ✅ Use useMemo for expensive calculations
function ExpensiveComponent({ data, filter }) {
  const expensiveValue = useMemo(() => {
    return data
      .filter(item => item.category === filter)
      .sort((a, b) => a.value - b.value)
      .slice(0, 100);
  }, [data, filter]);
  
  return <div>{/* Render expensive value */}</div>;
}

// ✅ Custom hooks for reusable logic
function useForm(initialValues, validate) {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});

  const handleChange = useCallback((name, value) => {
    setValues(prev => ({ ...prev, [name]: value }));
    
    if (touched[name]) {
      const fieldErrors = validate({ ...values, [name]: value });
      setErrors(prev => ({ ...prev, ...fieldErrors }));
    }
  }, [values, touched, validate]);

  const handleBlur = useCallback((name) => {
    setTouched(prev => ({ ...prev, [name]: true }));
    const fieldErrors = validate(values);
    setErrors(prev => ({ ...prev, ...fieldErrors }));
  }, [values, validate]);

  const handleSubmit = useCallback((onSubmit) => {
    const formErrors = validate(values);
    setErrors(formErrors);
    setTouched(Object.keys(values).reduce((acc, key) => ({ ...acc, [key]: true }), {}));
    
    if (Object.keys(formErrors).length === 0) {
      onSubmit(values);
    }
  }, [values, validate]);

  return {
    values,
    errors,
    touched,
    handleChange,
    handleBlur,
    handleSubmit
  };
}
```

## Performance Hooks

### React.memo, useMemo, and useCallback

```javascript
import { memo, useMemo, useCallback, useState } from 'react';

// Child component that should only re-render when props change
const ExpensiveChild = memo(({ items, onItemClick, multiplier }) => {
  console.log('ExpensiveChild rendered');
  
  // Expensive calculation inside child component
  const processedItems = useMemo(() => {
    console.log('Processing items...');
    return items.map(item => ({
      ...item,
      processedValue: item.value * multiplier
    }));
  }, [items, multiplier]);

  return (
    <div>
      {processedItems.map(item => (
        <div key={item.id} onClick={() => onItemClick(item.id)}>
          {item.name}: {item.processedValue}
        </div>
      ))}
    </div>
  );
});

function ParentComponent() {
  const [items, setItems] = useState([
    { id: 1, name: 'Item 1', value: 10 },
    { id: 2, name: 'Item 2', value: 20 }
  ]);
  const [multiplier, setMultiplier] = useState(2);
  const [otherState, setOtherState] = useState(0);

  // ❌ Without useCallback, this function is recreated on every render
  // causing ExpensiveChild to re-render even when items haven't changed
  // const handleItemClick = (id) => {
  //   console.log('Item clicked:', id);
  // };

  // ✅ With useCallback, function reference stays the same
  const handleItemClick = useCallback((id) => {
    console.log('Item clicked:', id);
    // Update items if needed
    setItems(prev => prev.map(item => 
      item.id === id ? { ...item, clicked: true } : item
    ));
  }, []); // No dependencies, function never changes

  return (
    <div>
      <button onClick={() => setMultiplier(multiplier + 1)}>
        Increase Multiplier: {multiplier}
      </button>
      
      <button onClick={() => setOtherState(otherState + 1)}>
        Other State: {otherState}
      </button>

      <ExpensiveChild 
        items={items}
        onItemClick={handleItemClick}
        multiplier={multiplier}
      />
    </div>
  );
}
```

## Common Hook Recipes

### Form Handling
```javascript
function useFormValidation(initialState, validate) {
  const [values, setValues] = useState(initialState);
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setValues({ ...values, [name]: value });
  };

  const handleSubmit = async (event, submitCallback) => {
    event.preventDefault();
    const validationErrors = validate(values);
    setErrors(validationErrors);
    setIsSubmitting(true);

    if (Object.keys(validationErrors).length === 0) {
      try {
        await submitCallback(values);
      } catch (error) {
        setErrors({ submit: error.message });
      }
    }
    setIsSubmitting(false);
  };

  return {
    handleChange,
    handleSubmit,
    values,
    errors,
    isSubmitting
  };
}

// Usage
function LoginForm() {
  const validate = (values) => {
    const errors = {};
    if (!values.email) errors.email = 'Email is required';
    if (!values.password) errors.password = 'Password is required';
    return errors;
  };

  const { handleChange, handleSubmit, values, errors, isSubmitting } = 
    useFormValidation({ email: '', password: '' }, validate);

  const login = async (userData) => {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });
    
    if (!response.ok) {
      throw new Error('Login failed');
    }
  };

  return (
    <form onSubmit={(e) => handleSubmit(e, login)}>
      <input
        name="email"
        type="email"
        value={values.email}
        onChange={handleChange}
        placeholder="Email"
      />
      {errors.email && <span>{errors.email}</span>}
      
      <input
        name="password"
        type="password"
        value={values.password}
        onChange={handleChange}
        placeholder="Password"
      />
      {errors.password && <span>{errors.password}</span>}
      
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Logging in...' : 'Login'}
      </button>
      
      {errors.submit && <span>{errors.submit}</span>}
    </form>
  );
}
```

### API State Management
```javascript
function useApi(url, options = {}) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null
  });

  const fetchData = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await fetch(url, options);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setState({ data, loading: false, error: null });
    } catch (error) {
      setState({ data: null, loading: false, error: error.message });
    }
  }, [url, JSON.stringify(options)]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { ...state, refetch: fetchData };
}

// Usage
function UserList() {
  const { data: users, loading, error, refetch } = useApi('/api/users');

  if (loading) return <div>Loading users...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <button onClick={refetch}>Refresh</button>
      <ul>
        {users?.map(user => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

---

*Continue to: [04-react-routing.md](./04-react-routing.md)*
