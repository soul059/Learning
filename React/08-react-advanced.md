# React Advanced Patterns

## Table of Contents
- [Higher-Order Components (HOCs)](#higher-order-components-hocs)
- [Render Props Pattern](#render-props-pattern)
- [Compound Components](#compound-components)
- [State Reducer Pattern](#state-reducer-pattern)
- [Control Props Pattern](#control-props-pattern)
- [Custom Hooks Patterns](#custom-hooks-patterns)
- [Error Boundaries](#error-boundaries)
- [Portals](#portals)
- [Refs and Forwarding](#refs-and-forwarding)
- [Concurrent Features](#concurrent-features)

## Higher-Order Components (HOCs)

### Basic HOC Pattern
```javascript
import React, { Component } from 'react';

// HOC that adds loading functionality
function withLoading(WrappedComponent) {
  return class WithLoadingComponent extends Component {
    constructor(props) {
      super(props);
      this.state = { isLoading: false };
    }

    setLoading = (isLoading) => {
      this.setState({ isLoading });
    };

    render() {
      const { isLoading } = this.state;
      
      if (isLoading) {
        return <div>Loading...</div>;
      }

      return (
        <WrappedComponent
          {...this.props}
          setLoading={this.setLoading}
        />
      );
    }
  };
}

// Usage
function UserProfile({ user, setLoading }) {
  const handleRefresh = async () => {
    setLoading(true);
    await fetchUserData();
    setLoading(false);
  };

  return (
    <div>
      <h1>{user.name}</h1>
      <button onClick={handleRefresh}>Refresh</button>
    </div>
  );
}

const UserProfileWithLoading = withLoading(UserProfile);
```

### Authentication HOC
```javascript
import React from 'react';
import { useAuth } from './hooks/useAuth';

function withAuth(WrappedComponent, { requireAuth = true, roles = [] } = {}) {
  function WithAuthComponent(props) {
    const { user, isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
      return <div>Checking authentication...</div>;
    }

    if (requireAuth && !isAuthenticated) {
      return <div>Please log in to access this page</div>;
    }

    if (roles.length > 0 && !roles.some(role => user?.roles?.includes(role))) {
      return <div>You don't have permission to access this page</div>;
    }

    return <WrappedComponent {...props} user={user} />;
  }

  WithAuthComponent.displayName = `withAuth(${WrappedComponent.displayName || WrappedComponent.name})`;
  
  return WithAuthComponent;
}

// Usage
const AdminPanel = withAuth(AdminPanelComponent, { 
  requireAuth: true, 
  roles: ['admin'] 
});

const PublicComponent = withAuth(PublicComponentBase, { 
  requireAuth: false 
});
```

### Data Fetching HOC
```javascript
import React, { useState, useEffect } from 'react';

function withDataFetching(WrappedComponent, fetchFunction) {
  return function WithDataFetchingComponent(props) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
      const fetchData = async () => {
        try {
          setLoading(true);
          const result = await fetchFunction(props);
          setData(result);
        } catch (err) {
          setError(err.message);
        } finally {
          setLoading(false);
        }
      };

      fetchData();
    }, [props.id]); // Re-fetch when ID changes

    const retry = () => {
      setError(null);
      fetchData();
    };

    return (
      <WrappedComponent
        {...props}
        data={data}
        loading={loading}
        error={error}
        retry={retry}
      />
    );
  };
}

// Usage
const fetchUser = async (props) => {
  const response = await fetch(`/api/users/${props.userId}`);
  return response.json();
};

const UserComponent = ({ data: user, loading, error, retry }) => {
  if (loading) return <div>Loading user...</div>;
  if (error) return <div>Error: {error} <button onClick={retry}>Retry</button></div>;
  
  return <div>Welcome, {user.name}!</div>;
};

const UserWithData = withDataFetching(UserComponent, fetchUser);
```

### Composing Multiple HOCs
```javascript
import { compose } from 'redux'; // or create your own compose function

function compose(...funcs) {
  if (funcs.length === 0) {
    return arg => arg;
  }

  if (funcs.length === 1) {
    return funcs[0];
  }

  return funcs.reduce((a, b) => (...args) => a(b(...args)));
}

// Multiple HOCs
const enhance = compose(
  withAuth({ requireAuth: true }),
  withLoading,
  withDataFetching(fetchUserProfile),
  withErrorBoundary
);

const EnhancedUserProfile = enhance(UserProfile);

// Or using individual HOCs
const EnhancedUserProfile2 = withAuth(
  withLoading(
    withDataFetching(
      withErrorBoundary(UserProfile),
      fetchUserProfile
    )
  ),
  { requireAuth: true }
);
```

## Render Props Pattern

### Basic Render Props
```javascript
import React, { useState } from 'react';

// Mouse tracker component using render props
function MouseTracker({ render }) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  const handleMouseMove = (event) => {
    setMousePosition({
      x: event.clientX,
      y: event.clientY,
    });
  };

  return (
    <div style={{ height: '100vh' }} onMouseMove={handleMouseMove}>
      {render(mousePosition)}
    </div>
  );
}

// Usage
function App() {
  return (
    <MouseTracker
      render={({ x, y }) => (
        <div>
          <h1>Mouse position:</h1>
          <p>X: {x}, Y: {y}</p>
        </div>
      )}
    />
  );
}

// Alternative using children function
function MouseTracker2({ children }) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  const handleMouseMove = (event) => {
    setMousePosition({
      x: event.clientX,
      y: event.clientY,
    });
  };

  return (
    <div style={{ height: '100vh' }} onMouseMove={handleMouseMove}>
      {children(mousePosition)}
    </div>
  );
}

// Usage with children
function App2() {
  return (
    <MouseTracker2>
      {({ x, y }) => (
        <div>Mouse at ({x}, {y})</div>
      )}
    </MouseTracker2>
  );
}
```

### Data Provider with Render Props
```javascript
import React, { useState, useEffect } from 'react';

function DataProvider({ url, children, render }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch');
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [url]);

  const renderFunction = render || children;
  return renderFunction({ data, loading, error });
}

// Usage examples
function UsersList() {
  return (
    <DataProvider url="/api/users">
      {({ data: users, loading, error }) => {
        if (loading) return <div>Loading users...</div>;
        if (error) return <div>Error: {error}</div>;
        
        return (
          <ul>
            {users.map(user => (
              <li key={user.id}>{user.name}</li>
            ))}
          </ul>
        );
      }}
    </DataProvider>
  );
}

function UserProfile({ userId }) {
  return (
    <DataProvider url={`/api/users/${userId}`}>
      {({ data: user, loading, error }) => {
        if (loading) return <div>Loading profile...</div>;
        if (error) return <div>Error: {error}</div>;
        
        return (
          <div>
            <h1>{user.name}</h1>
            <p>{user.email}</p>
          </div>
        );
      }}
    </DataProvider>
  );
}
```

### Form Render Props
```javascript
import React, { useState } from 'react';

function Form({ onSubmit, children }) {
  const [values, setValues] = useState({});
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});

  const setValue = (name, value) => {
    setValues(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  const setError = (name, error) => {
    setErrors(prev => ({ ...prev, [name]: error }));
  };

  const setTouched = (name) => {
    setTouched(prev => ({ ...prev, [name]: true }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(values, { setError });
  };

  return (
    <form onSubmit={handleSubmit}>
      {children({
        values,
        errors,
        touched,
        setValue,
        setError,
        setTouched,
      })}
    </form>
  );
}

// Usage
function LoginForm() {
  const handleSubmit = async (values, { setError }) => {
    try {
      await login(values);
    } catch (error) {
      setError('general', error.message);
    }
  };

  return (
    <Form onSubmit={handleSubmit}>
      {({ values, errors, setValue, setTouched }) => (
        <>
          <input
            type="email"
            value={values.email || ''}
            onChange={(e) => setValue('email', e.target.value)}
            onBlur={() => setTouched('email')}
            placeholder="Email"
          />
          {errors.email && <span>{errors.email}</span>}
          
          <input
            type="password"
            value={values.password || ''}
            onChange={(e) => setValue('password', e.target.value)}
            onBlur={() => setTouched('password')}
            placeholder="Password"
          />
          {errors.password && <span>{errors.password}</span>}
          
          {errors.general && <div>{errors.general}</div>}
          
          <button type="submit">Login</button>
        </>
      )}
    </Form>
  );
}
```

## Compound Components

### Basic Compound Component
```javascript
import React, { createContext, useContext, useState } from 'react';

// Context for compound component
const TabsContext = createContext();

// Main Tabs component
function Tabs({ defaultTab, children }) {
  const [activeTab, setActiveTab] = useState(defaultTab);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
}

// Tab List component
function TabList({ children }) {
  return <div className="tab-list">{children}</div>;
}

// Individual Tab component
function Tab({ id, children }) {
  const { activeTab, setActiveTab } = useContext(TabsContext);
  const isActive = activeTab === id;

  return (
    <button
      className={`tab ${isActive ? 'active' : ''}`}
      onClick={() => setActiveTab(id)}
    >
      {children}
    </button>
  );
}

// Tab Panels container
function TabPanels({ children }) {
  return <div className="tab-panels">{children}</div>;
}

// Individual Tab Panel
function TabPanel({ id, children }) {
  const { activeTab } = useContext(TabsContext);
  
  if (activeTab !== id) return null;

  return <div className="tab-panel">{children}</div>;
}

// Attach components to main Tabs component
Tabs.List = TabList;
Tabs.Tab = Tab;
Tabs.Panels = TabPanels;
Tabs.Panel = TabPanel;

// Usage
function App() {
  return (
    <Tabs defaultTab="tab1">
      <Tabs.List>
        <Tabs.Tab id="tab1">Tab 1</Tabs.Tab>
        <Tabs.Tab id="tab2">Tab 2</Tabs.Tab>
        <Tabs.Tab id="tab3">Tab 3</Tabs.Tab>
      </Tabs.List>
      
      <Tabs.Panels>
        <Tabs.Panel id="tab1">
          <h2>Content for Tab 1</h2>
          <p>This is the first tab's content.</p>
        </Tabs.Panel>
        
        <Tabs.Panel id="tab2">
          <h2>Content for Tab 2</h2>
          <p>This is the second tab's content.</p>
        </Tabs.Panel>
        
        <Tabs.Panel id="tab3">
          <h2>Content for Tab 3</h2>
          <p>This is the third tab's content.</p>
        </Tabs.Panel>
      </Tabs.Panels>
    </Tabs>
  );
}
```

### Modal Compound Component
```javascript
import React, { createContext, useContext, useState } from 'react';
import ReactDOM from 'react-dom';

const ModalContext = createContext();

function Modal({ children }) {
  const [isOpen, setIsOpen] = useState(false);

  const open = () => setIsOpen(true);
  const close = () => setIsOpen(false);

  return (
    <ModalContext.Provider value={{ isOpen, open, close }}>
      {children}
    </ModalContext.Provider>
  );
}

function ModalTrigger({ children }) {
  const { open } = useContext(ModalContext);
  
  return React.cloneElement(children, { onClick: open });
}

function ModalContent({ children }) {
  const { isOpen, close } = useContext(ModalContext);

  if (!isOpen) return null;

  return ReactDOM.createPortal(
    <div className="modal-overlay" onClick={close}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>,
    document.body
  );
}

function ModalHeader({ children }) {
  return <div className="modal-header">{children}</div>;
}

function ModalBody({ children }) {
  return <div className="modal-body">{children}</div>;
}

function ModalFooter({ children }) {
  return <div className="modal-footer">{children}</div>;
}

function ModalCloseButton({ children }) {
  const { close } = useContext(ModalContext);
  
  return (
    <button className="modal-close" onClick={close}>
      {children || '×'}
    </button>
  );
}

// Attach components
Modal.Trigger = ModalTrigger;
Modal.Content = ModalContent;
Modal.Header = ModalHeader;
Modal.Body = ModalBody;
Modal.Footer = ModalFooter;
Modal.CloseButton = ModalCloseButton;

// Usage
function App() {
  return (
    <Modal>
      <Modal.Trigger>
        <button>Open Modal</button>
      </Modal.Trigger>
      
      <Modal.Content>
        <Modal.Header>
          <h2>Modal Title</h2>
          <Modal.CloseButton />
        </Modal.Header>
        
        <Modal.Body>
          <p>This is the modal content.</p>
        </Modal.Body>
        
        <Modal.Footer>
          <Modal.CloseButton>
            <button>Cancel</button>
          </Modal.CloseButton>
          <button>Save</button>
        </Modal.Footer>
      </Modal.Content>
    </Modal>
  );
}
```

## State Reducer Pattern

### Basic State Reducer
```javascript
import React, { useReducer } from 'react';

// Counter with reducer pattern
function counterReducer(state, action) {
  switch (action.type) {
    case 'INCREMENT':
      return { count: state.count + 1 };
    case 'DECREMENT':
      return { count: state.count - 1 };
    case 'RESET':
      return { count: 0 };
    case 'SET':
      return { count: action.payload };
    default:
      throw new Error(`Unhandled action type: ${action.type}`);
  }
}

function useCounter(initialValue = 0, reducer = counterReducer) {
  const [state, dispatch] = useReducer(reducer, { count: initialValue });

  const increment = () => dispatch({ type: 'INCREMENT' });
  const decrement = () => dispatch({ type: 'DECREMENT' });
  const reset = () => dispatch({ type: 'RESET' });
  const set = (value) => dispatch({ type: 'SET', payload: value });

  return {
    count: state.count,
    increment,
    decrement,
    reset,
    set,
    dispatch, // Allow custom dispatches
  };
}

// Custom reducer that extends default behavior
function customCounterReducer(state, action) {
  switch (action.type) {
    case 'DOUBLE':
      return { count: state.count * 2 };
    case 'HALF':
      return { count: Math.floor(state.count / 2) };
    default:
      return counterReducer(state, action);
  }
}

// Usage
function Counter() {
  const { count, increment, decrement, reset, dispatch } = useCounter(0, customCounterReducer);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>+</button>
      <button onClick={decrement}>-</button>
      <button onClick={reset}>Reset</button>
      <button onClick={() => dispatch({ type: 'DOUBLE' })}>Double</button>
      <button onClick={() => dispatch({ type: 'HALF' })}>Half</button>
    </div>
  );
}
```

### Complex State Reducer (Todo List)
```javascript
import React, { useReducer } from 'react';

function todoReducer(state, action) {
  switch (action.type) {
    case 'ADD_TODO':
      return {
        ...state,
        todos: [
          ...state.todos,
          {
            id: Date.now(),
            text: action.payload,
            completed: false,
          },
        ],
      };
    
    case 'TOGGLE_TODO':
      return {
        ...state,
        todos: state.todos.map(todo =>
          todo.id === action.payload
            ? { ...todo, completed: !todo.completed }
            : todo
        ),
      };
    
    case 'DELETE_TODO':
      return {
        ...state,
        todos: state.todos.filter(todo => todo.id !== action.payload),
      };
    
    case 'SET_FILTER':
      return {
        ...state,
        filter: action.payload,
      };
    
    case 'CLEAR_COMPLETED':
      return {
        ...state,
        todos: state.todos.filter(todo => !todo.completed),
      };
    
    default:
      return state;
  }
}

function useTodos(initialTodos = [], reducer = todoReducer) {
  const [state, dispatch] = useReducer(reducer, {
    todos: initialTodos,
    filter: 'ALL', // ALL, ACTIVE, COMPLETED
  });

  const addTodo = (text) => {
    dispatch({ type: 'ADD_TODO', payload: text });
  };

  const toggleTodo = (id) => {
    dispatch({ type: 'TOGGLE_TODO', payload: id });
  };

  const deleteTodo = (id) => {
    dispatch({ type: 'DELETE_TODO', payload: id });
  };

  const setFilter = (filter) => {
    dispatch({ type: 'SET_FILTER', payload: filter });
  };

  const clearCompleted = () => {
    dispatch({ type: 'CLEAR_COMPLETED' });
  };

  const filteredTodos = state.todos.filter(todo => {
    switch (state.filter) {
      case 'ACTIVE':
        return !todo.completed;
      case 'COMPLETED':
        return todo.completed;
      default:
        return true;
    }
  });

  return {
    todos: filteredTodos,
    filter: state.filter,
    addTodo,
    toggleTodo,
    deleteTodo,
    setFilter,
    clearCompleted,
    dispatch,
  };
}

// Enhanced reducer with undo/redo
function enhancedTodoReducer(state, action) {
  switch (action.type) {
    case 'UNDO':
      if (state.past.length > 0) {
        const previous = state.past[state.past.length - 1];
        const newPast = state.past.slice(0, state.past.length - 1);
        return {
          past: newPast,
          present: previous,
          future: [state.present, ...state.future],
        };
      }
      return state;
    
    case 'REDO':
      if (state.future.length > 0) {
        const next = state.future[0];
        const newFuture = state.future.slice(1);
        return {
          past: [...state.past, state.present],
          present: next,
          future: newFuture,
        };
      }
      return state;
    
    default:
      const newPresent = todoReducer(state.present, action);
      if (newPresent === state.present) {
        return state;
      }
      return {
        past: [...state.past, state.present],
        present: newPresent,
        future: [],
      };
  }
}
```

## Control Props Pattern

### Controlled Component Pattern
```javascript
import React, { useState } from 'react';

function Switch({ 
  checked, 
  defaultChecked = false, 
  onChange 
}) {
  // Determine if component is controlled or uncontrolled
  const isControlled = checked !== undefined;
  const [internalChecked, setInternalChecked] = useState(defaultChecked);
  
  const checkedValue = isControlled ? checked : internalChecked;

  const handleChange = (event) => {
    const newChecked = event.target.checked;
    
    if (!isControlled) {
      setInternalChecked(newChecked);
    }
    
    onChange?.(newChecked);
  };

  return (
    <label>
      <input
        type="checkbox"
        checked={checkedValue}
        onChange={handleChange}
      />
      <span className="slider"></span>
    </label>
  );
}

// Usage - Uncontrolled
function UncontrolledExample() {
  return (
    <Switch
      defaultChecked={true}
      onChange={(checked) => console.log('Switch is', checked)}
    />
  );
}

// Usage - Controlled
function ControlledExample() {
  const [isOn, setIsOn] = useState(false);

  return (
    <Switch
      checked={isOn}
      onChange={setIsOn}
    />
  );
}
```

### Advanced Control Props with State Reducer
```javascript
import React, { useState, useReducer } from 'react';

function toggleReducer(state, action) {
  switch (action.type) {
    case 'TOGGLE':
      return { on: !state.on };
    case 'RESET':
      return { on: false };
    default:
      return state;
  }
}

function useToggle({
  initialOn = false,
  reducer = toggleReducer,
  on: controlledOn,
  onChange,
}) {
  const [state, dispatch] = useReducer(reducer, { on: initialOn });
  const isControlled = controlledOn !== undefined;
  const on = isControlled ? controlledOn : state.on;

  const dispatchWithOnChange = (action) => {
    if (!isControlled) {
      dispatch(action);
    }
    
    const newState = reducer(state, action);
    onChange?.(newState, action);
  };

  const toggle = () => dispatchWithOnChange({ type: 'TOGGLE' });
  const reset = () => dispatchWithOnChange({ type: 'RESET' });

  return {
    on,
    toggle,
    reset,
    dispatch: dispatchWithOnChange,
  };
}

function Toggle({ 
  on: controlledOn, 
  onChange, 
  reducer,
  children 
}) {
  const { on, toggle, reset } = useToggle({
    on: controlledOn,
    onChange,
    reducer,
  });

  return children({ on, toggle, reset });
}

// Custom reducer that prevents toggling off
function noToggleOffReducer(state, action) {
  if (action.type === 'TOGGLE' && state.on) {
    return state; // Prevent toggling off
  }
  return toggleReducer(state, action);
}

// Usage
function App() {
  const [bothOn, setBothOn] = useState(false);

  return (
    <div>
      {/* Uncontrolled */}
      <Toggle>
        {({ on, toggle, reset }) => (
          <div>
            <Switch on={on} onClick={toggle} />
            <button onClick={reset}>Reset</button>
          </div>
        )}
      </Toggle>

      {/* Controlled */}
      <Toggle
        on={bothOn}
        onChange={(state) => setBothOn(state.on)}
      >
        {({ on, toggle }) => (
          <Switch on={on} onClick={toggle} />
        )}
      </Toggle>

      {/* With custom reducer */}
      <Toggle reducer={noToggleOffReducer}>
        {({ on, toggle, reset }) => (
          <div>
            <Switch on={on} onClick={toggle} />
            <button onClick={reset}>Reset</button>
            <p>This switch can't be turned off, only reset!</p>
          </div>
        )}
      </Toggle>
    </div>
  );
}
```

## Custom Hooks Patterns

### Async Data Fetching Hook
```javascript
import { useState, useEffect, useRef, useCallback } from 'react';

function useAsyncData(asyncFunction, dependencies = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const cancelRef = useRef();

  const execute = useCallback(async (...args) => {
    try {
      setLoading(true);
      setError(null);
      
      // Cancel previous request
      if (cancelRef.current) {
        cancelRef.current();
      }

      // Create cancellation token
      let cancelled = false;
      cancelRef.current = () => {
        cancelled = true;
      };

      const result = await asyncFunction(...args);
      
      if (!cancelled) {
        setData(result);
      }
    } catch (err) {
      if (!cancelled) {
        setError(err);
      }
    } finally {
      if (!cancelled) {
        setLoading(false);
      }
    }
  }, dependencies);

  useEffect(() => {
    execute();
    
    return () => {
      if (cancelRef.current) {
        cancelRef.current();
      }
    };
  }, [execute]);

  const retry = () => execute();

  return { data, loading, error, retry, execute };
}

// Usage
function UserProfile({ userId }) {
  const { 
    data: user, 
    loading, 
    error, 
    retry 
  } = useAsyncData(
    async () => {
      const response = await fetch(`/api/users/${userId}`);
      return response.json();
    }, 
    [userId]
  );

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message} <button onClick={retry}>Retry</button></div>;

  return <div>Welcome, {user.name}!</div>;
}
```

### Local Storage Hook
```javascript
import { useState, useEffect } from 'react';

function useLocalStorage(key, initialValue) {
  // Get value from localStorage or use initial value
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  // Update localStorage when value changes
  const setValue = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error);
    }
  };

  // Listen for changes in other tabs/windows
  useEffect(() => {
    const handleStorageChange = (e) => {
      if (e.key === key && e.newValue !== null) {
        try {
          setStoredValue(JSON.parse(e.newValue));
        } catch (error) {
          console.warn(`Error parsing localStorage key "${key}":`, error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [key]);

  return [storedValue, setValue];
}

// Enhanced version with options
function useLocalStorageAdvanced(key, initialValue, options = {}) {
  const {
    serialize = JSON.stringify,
    deserialize = JSON.parse,
    validator,
    onError,
  } = options;

  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      if (item !== null) {
        const parsed = deserialize(item);
        if (validator && !validator(parsed)) {
          throw new Error('Validation failed');
        }
        return parsed;
      }
      return initialValue;
    } catch (error) {
      onError?.(error);
      return initialValue;
    }
  });

  const setValue = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      
      if (validator && !validator(valueToStore)) {
        throw new Error('Validation failed');
      }
      
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, serialize(valueToStore));
    } catch (error) {
      onError?.(error);
    }
  };

  const removeValue = () => {
    try {
      setStoredValue(initialValue);
      window.localStorage.removeItem(key);
    } catch (error) {
      onError?.(error);
    }
  };

  return [storedValue, setValue, removeValue];
}
```

### Intersection Observer Hook
```javascript
import { useEffect, useRef, useState } from 'react';

function useIntersectionObserver(options = {}) {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [entry, setEntry] = useState(null);
  const targetRef = useRef(null);

  useEffect(() => {
    const target = targetRef.current;
    if (!target) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
        setEntry(entry);
      },
      options
    );

    observer.observe(target);

    return () => {
      observer.unobserve(target);
    };
  }, [options.threshold, options.root, options.rootMargin]);

  return [targetRef, isIntersecting, entry];
}

// Multiple elements version
function useIntersectionObserverMultiple(options = {}) {
  const [entries, setEntries] = useState(new Map());
  const observerRef = useRef(null);

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (observedEntries) => {
        setEntries(prev => {
          const newEntries = new Map(prev);
          observedEntries.forEach(entry => {
            newEntries.set(entry.target, entry);
          });
          return newEntries;
        });
      },
      options
    );

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [options.threshold, options.root, options.rootMargin]);

  const observe = useCallback((element) => {
    if (observerRef.current && element) {
      observerRef.current.observe(element);
    }
  }, []);

  const unobserve = useCallback((element) => {
    if (observerRef.current && element) {
      observerRef.current.unobserve(element);
      setEntries(prev => {
        const newEntries = new Map(prev);
        newEntries.delete(element);
        return newEntries;
      });
    }
  }, []);

  return { entries, observe, unobserve };
}

// Usage
function LazyImage({ src, alt, ...props }) {
  const [imgRef, isVisible] = useIntersectionObserver({
    threshold: 0.1,
    rootMargin: '50px'
  });

  return (
    <div ref={imgRef} {...props}>
      {isVisible ? (
        <img src={src} alt={alt} />
      ) : (
        <div>Loading...</div>
      )}
    </div>
  );
}
```

## Error Boundaries

### Basic Error Boundary
```javascript
import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error to reporting service
    this.setState({
      error,
      errorInfo
    });
    
    // Log to external service
    this.logErrorToService(error, errorInfo);
  }

  logErrorToService = (error, errorInfo) => {
    // Example: Send to error reporting service
    console.error('Error caught by boundary:', error, errorInfo);
    
    // In real app, send to service like Sentry
    // Sentry.captureException(error, { extra: errorInfo });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div style={{ padding: '20px', backgroundColor: '#ffebee' }}>
          <h2>Something went wrong!</h2>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            <summary>Error details</summary>
            <p>{this.state.error && this.state.error.toString()}</p>
            <p>{this.state.errorInfo.componentStack}</p>
          </details>
          <button onClick={() => window.location.reload()}>
            Reload page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Usage
function App() {
  return (
    <ErrorBoundary>
      <Header />
      <ErrorBoundary fallback={<div>Error in main content</div>}>
        <MainContent />
      </ErrorBoundary>
      <Footer />
    </ErrorBoundary>
  );
}
```

### Hook-based Error Boundary
```javascript
import { useState, useEffect } from 'react';

function useErrorHandler() {
  const [error, setError] = useState(null);

  const resetError = () => setError(null);

  const captureError = (error, errorInfo) => {
    setError({ error, errorInfo });
    
    // Log to external service
    console.error('Error captured:', error, errorInfo);
  };

  return { error, resetError, captureError };
}

function withErrorBoundary(Component, fallback) {
  return function WrappedComponent(props) {
    return (
      <ErrorBoundary fallback={fallback}>
        <Component {...props} />
      </ErrorBoundary>
    );
  };
}

// Async error handling
function useAsyncError() {
  const [, setError] = useState();
  
  return useCallback((error) => {
    setError(() => {
      throw error;
    });
  }, [setError]);
}

function AsyncComponent() {
  const throwAsyncError = useAsyncError();

  const handleAsyncError = async () => {
    try {
      await riskyAsyncOperation();
    } catch (error) {
      throwAsyncError(error);
    }
  };

  return (
    <button onClick={handleAsyncError}>
      Trigger async error
    </button>
  );
}
```

## Portals

### Basic Portal Usage
```javascript
import React from 'react';
import ReactDOM from 'react-dom';

function Modal({ isOpen, onClose, children }) {
  if (!isOpen) return null;

  return ReactDOM.createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        {children}
      </div>
    </div>,
    document.body
  );
}

// Tooltip portal
function Tooltip({ children, content, position = 'top' }) {
  const [isVisible, setIsVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const triggerRef = useRef();

  const showTooltip = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setTooltipPosition({
      x: rect.left + rect.width / 2,
      y: rect.top - 10,
    });
    setIsVisible(true);
  };

  const hideTooltip = () => {
    setIsVisible(false);
  };

  return (
    <>
      <span
        ref={triggerRef}
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
      >
        {children}
      </span>
      
      {isVisible && ReactDOM.createPortal(
        <div
          className={`tooltip tooltip-${position}`}
          style={{
            position: 'fixed',
            left: tooltipPosition.x,
            top: tooltipPosition.y,
            transform: 'translateX(-50%) translateY(-100%)',
            zIndex: 9999,
          }}
        >
          {content}
        </div>,
        document.body
      )}
    </>
  );
}
```

### Portal Container Management
```javascript
import { useEffect, useRef } from 'react';

function usePortal(id) {
  const rootElemRef = useRef(null);

  useEffect(() => {
    // Look for existing portal
    const existingParent = document.querySelector(`#${id}`);
    
    // Create new portal if none exists
    const parentElem = existingParent || createRootElement(id);
    
    // Add to DOM if new
    if (!existingParent) {
      addRootElement(parentElem);
    }
    
    // Add element to portal
    parentElem.appendChild(rootElemRef.current);
    
    return () => {
      rootElemRef.current.remove();
      if (parentElem.childNodes.length === 0) {
        parentElem.remove();
      }
    };
  }, [id]);

  function createRootElement(id) {
    const rootContainer = document.createElement('div');
    rootContainer.setAttribute('id', id);
    rootContainer.setAttribute('data-portal', 'true');
    return rootContainer;
  }

  function addRootElement(rootElem) {
    document.body.insertBefore(
      rootElem,
      document.body.lastElementChild.nextElementSibling
    );
  }

  function getRootElem() {
    if (!rootElemRef.current) {
      rootElemRef.current = document.createElement('div');
    }
    return rootElemRef.current;
  }

  return getRootElem();
}

// Portal component using the hook
function Portal({ id, children }) {
  const target = usePortal(id);
  
  return ReactDOM.createPortal(children, target);
}

// Usage
function App() {
  const [showModal, setShowModal] = useState(false);

  return (
    <div>
      <button onClick={() => setShowModal(true)}>
        Open Modal
      </button>
      
      <Portal id="modal-root">
        <Modal isOpen={showModal} onClose={() => setShowModal(false)}>
          <h2>Portal Modal</h2>
          <p>This modal is rendered in a portal!</p>
        </Modal>
      </Portal>
    </div>
  );
}
```

## Refs and Forwarding

### Ref Forwarding
```javascript
import React, { forwardRef, useImperativeHandle } from 'react';

// Basic ref forwarding
const FancyButton = forwardRef((props, ref) => (
  <button ref={ref} className="fancy-button">
    {props.children}
  </button>
));

// Custom imperative handle
const TextInput = forwardRef((props, ref) => {
  const inputRef = useRef();

  useImperativeHandle(ref, () => ({
    focus: () => {
      inputRef.current.focus();
    },
    blur: () => {
      inputRef.current.blur();
    },
    setValue: (value) => {
      inputRef.current.value = value;
    },
    getValue: () => {
      return inputRef.current.value;
    },
  }));

  return <input ref={inputRef} {...props} />;
});

// Usage
function App() {
  const buttonRef = useRef();
  const inputRef = useRef();

  const handleClick = () => {
    // Focus the button
    buttonRef.current.focus();
    
    // Use custom methods
    inputRef.current.setValue('Hello World');
    inputRef.current.focus();
  };

  return (
    <div>
      <TextInput ref={inputRef} placeholder="Enter text" />
      <FancyButton ref={buttonRef} onClick={handleClick}>
        Click me
      </FancyButton>
    </div>
  );
}
```

### Complex Ref Patterns
```javascript
import React, { useRef, useCallback } from 'react';

// Callback refs
function CallbackRefExample() {
  const [height, setHeight] = useState(0);

  const measuredRef = useCallback(node => {
    if (node !== null) {
      setHeight(node.getBoundingClientRect().height);
    }
  }, []);

  return (
    <div>
      <h1 ref={measuredRef}>Hello, world</h1>
      <h2>The above header is {Math.round(height)}px tall</h2>
    </div>
  );
}

// Ref with multiple elements
function useRefs() {
  const refs = useRef(new Map());

  const setRef = useCallback((element, key) => {
    if (element) {
      refs.current.set(key, element);
    } else {
      refs.current.delete(key);
    }
  }, []);

  const getRef = useCallback((key) => {
    return refs.current.get(key);
  }, []);

  return [setRef, getRef, refs.current];
}

function MultipleRefsExample() {
  const [setRef, getRef] = useRefs();
  const [focusedItem, setFocusedItem] = useState(null);

  const items = ['first', 'second', 'third'];

  const focusItem = (key) => {
    const element = getRef(key);
    if (element) {
      element.focus();
      setFocusedItem(key);
    }
  };

  return (
    <div>
      {items.map(item => (
        <input
          key={item}
          ref={el => setRef(el, item)}
          placeholder={`Input ${item}`}
          style={{
            margin: '5px',
            backgroundColor: focusedItem === item ? 'yellow' : 'white'
          }}
        />
      ))}
      
      {items.map(item => (
        <button key={item} onClick={() => focusItem(item)}>
          Focus {item}
        </button>
      ))}
    </div>
  );
}
```

## Concurrent Features

### Suspense and Lazy Loading
```javascript
import React, { Suspense, lazy, startTransition } from 'react';

// Lazy loaded components
const HeavyComponent = lazy(() => import('./HeavyComponent'));
const Chart = lazy(() => import('./Chart'));

function App() {
  const [tab, setTab] = useState('home');
  const [isPending, setIsPending] = useState(false);

  const switchTab = (newTab) => {
    setIsPending(true);
    startTransition(() => {
      setTab(newTab);
      setIsPending(false);
    });
  };

  return (
    <div>
      <nav>
        <button 
          onClick={() => switchTab('home')}
          disabled={isPending}
        >
          Home
        </button>
        <button 
          onClick={() => switchTab('chart')}
          disabled={isPending}
        >
          Chart {isPending && '(Loading...)'}
        </button>
      </nav>

      <Suspense fallback={<div>Loading content...</div>}>
        {tab === 'home' && <HomeComponent />}
        {tab === 'chart' && <Chart />}
      </Suspense>
    </div>
  );
}
```

### useDeferredValue
```javascript
import { useDeferredValue, useMemo, useState } from 'react';

function SearchResults({ query }) {
  const deferredQuery = useDeferredValue(query);
  const isStale = query !== deferredQuery;

  const results = useMemo(() => {
    // Expensive search operation
    return searchDatabase(deferredQuery);
  }, [deferredQuery]);

  return (
    <div style={{ opacity: isStale ? 0.5 : 1 }}>
      <h3>Search Results for "{deferredQuery}"</h3>
      {results.map(result => (
        <div key={result.id}>{result.title}</div>
      ))}
    </div>
  );
}

function SearchApp() {
  const [query, setQuery] = useState('');

  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search..."
      />
      <SearchResults query={query} />
    </div>
  );
}
```

### useTransition
```javascript
import { useTransition, useState } from 'react';

function TabContainer() {
  const [isPending, startTransition] = useTransition();
  const [tab, setTab] = useState('about');

  function selectTab(nextTab) {
    startTransition(() => {
      setTab(nextTab);
    });
  }

  return (
    <div>
      <TabButton
        isActive={tab === 'about'}
        onClick={() => selectTab('about')}
      >
        About
      </TabButton>
      <TabButton
        isActive={tab === 'posts'}
        onClick={() => selectTab('posts')}
      >
        Posts {isPending && ' (loading...)'}
      </TabButton>
      <TabButton
        isActive={tab === 'contact'}
        onClick={() => selectTab('contact')}
      >
        Contact
      </TabButton>
      
      <hr />
      
      {tab === 'about' && <AboutTab />}
      {tab === 'posts' && <PostsTab />}
      {tab === 'contact' && <ContactTab />}
    </div>
  );
}

function TabButton({ children, isActive, onClick }) {
  return (
    <button
      style={{
        backgroundColor: isActive ? 'blue' : 'gray',
        color: 'white',
        margin: '5px'
      }}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
```

---

*Continue to: [11-react-native-components.md](./11-react-native-components.md)*
