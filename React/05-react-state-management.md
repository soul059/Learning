# React State Management

## Table of Contents
- [State Management Overview](#state-management-overview)
- [Local State with useState](#local-state-with-usestate)
- [Context API](#context-api)
- [useReducer for Complex State](#usereducer-for-complex-state)
- [Redux Toolkit](#redux-toolkit)
- [Zustand](#zustand)
- [Jotai (Atomic State)](#jotai-atomic-state)
- [State Management Patterns](#state-management-patterns)
- [Best Practices](#best-practices)

## State Management Overview

State management is crucial for React applications. Choose the right tool based on your needs:

### When to Use Each Solution:

| Solution | Use Case | Complexity | Learning Curve |
|----------|----------|------------|----------------|
| **useState** | Component-level state | Low | Easy |
| **useReducer** | Complex component state | Medium | Easy |
| **Context API** | Theme, auth, small global state | Medium | Easy |
| **Redux Toolkit** | Large apps, complex state logic | High | Medium |
| **Zustand** | Simple global state | Low-Medium | Easy |
| **Jotai** | Atomic state management | Medium | Medium |

## Local State with useState

### Basic State Management
```javascript
import { useState } from 'react';

function ShoppingCart() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Add item to cart
  const addItem = (product) => {
    setItems(prevItems => {
      const existingItem = prevItems.find(item => item.id === product.id);
      
      if (existingItem) {
        return prevItems.map(item =>
          item.id === product.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }
      
      return [...prevItems, { ...product, quantity: 1 }];
    });
  };

  // Remove item from cart
  const removeItem = (productId) => {
    setItems(prevItems => prevItems.filter(item => item.id !== productId));
  };

  // Update quantity
  const updateQuantity = (productId, quantity) => {
    if (quantity <= 0) {
      removeItem(productId);
      return;
    }

    setItems(prevItems =>
      prevItems.map(item =>
        item.id === productId ? { ...item, quantity } : item
      )
    );
  };

  // Calculate totals
  const totalItems = items.reduce((sum, item) => sum + item.quantity, 0);
  const totalPrice = items.reduce((sum, item) => sum + (item.price * item.quantity), 0);

  return (
    <div>
      <h2>Shopping Cart ({totalItems} items)</h2>
      
      {error && <div className="error">{error}</div>}
      
      {items.length === 0 ? (
        <p>Your cart is empty</p>
      ) : (
        <div>
          {items.map(item => (
            <div key={item.id} className="cart-item">
              <h3>{item.name}</h3>
              <p>${item.price}</p>
              <div>
                <button onClick={() => updateQuantity(item.id, item.quantity - 1)}>
                  -
                </button>
                <span>{item.quantity}</span>
                <button onClick={() => updateQuantity(item.id, item.quantity + 1)}>
                  +
                </button>
              </div>
              <button onClick={() => removeItem(item.id)}>Remove</button>
            </div>
          ))}
          
          <div className="cart-total">
            Total: ${totalPrice.toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
}
```

### Custom State Hooks
```javascript
// Custom hook for form state
function useForm(initialValues, validate) {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (name, value) => {
    setValues(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const handleBlur = (name) => {
    setTouched(prev => ({ ...prev, [name]: true }));
    
    if (validate) {
      const fieldErrors = validate({ ...values, [name]: values[name] });
      setErrors(prev => ({ ...prev, [name]: fieldErrors[name] }));
    }
  };

  const handleSubmit = async (onSubmit) => {
    setIsSubmitting(true);
    
    if (validate) {
      const validationErrors = validate(values);
      setErrors(validationErrors);
      setTouched(Object.keys(values).reduce((acc, key) => ({ ...acc, [key]: true }), {}));
      
      if (Object.keys(validationErrors).length > 0) {
        setIsSubmitting(false);
        return;
      }
    }
    
    try {
      await onSubmit(values);
    } catch (error) {
      setErrors({ submit: error.message });
    } finally {
      setIsSubmitting(false);
    }
  };

  const reset = () => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  };

  return {
    values,
    errors,
    touched,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    reset
  };
}

// Usage
function ContactForm() {
  const validate = (values) => {
    const errors = {};
    
    if (!values.name) errors.name = 'Name is required';
    if (!values.email) errors.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(values.email)) errors.email = 'Email is invalid';
    if (!values.message) errors.message = 'Message is required';
    
    return errors;
  };

  const {
    values,
    errors,
    touched,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    reset
  } = useForm(
    { name: '', email: '', message: '' },
    validate
  );

  const onSubmit = async (formData) => {
    const response = await fetch('/api/contact', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });
    
    if (!response.ok) throw new Error('Failed to send message');
    
    alert('Message sent successfully!');
    reset();
  };

  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      handleSubmit(onSubmit);
    }}>
      <div>
        <input
          type="text"
          value={values.name}
          onChange={(e) => handleChange('name', e.target.value)}
          onBlur={() => handleBlur('name')}
          placeholder="Name"
        />
        {touched.name && errors.name && <span className="error">{errors.name}</span>}
      </div>

      <div>
        <input
          type="email"
          value={values.email}
          onChange={(e) => handleChange('email', e.target.value)}
          onBlur={() => handleBlur('email')}
          placeholder="Email"
        />
        {touched.email && errors.email && <span className="error">{errors.email}</span>}
      </div>

      <div>
        <textarea
          value={values.message}
          onChange={(e) => handleChange('message', e.target.value)}
          onBlur={() => handleBlur('message')}
          placeholder="Message"
        />
        {touched.message && errors.message && <span className="error">{errors.message}</span>}
      </div>

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Sending...' : 'Send Message'}
      </button>

      {errors.submit && <div className="error">{errors.submit}</div>}
    </form>
  );
}
```

## Context API

### Basic Context Setup
```javascript
import { createContext, useContext, useState, useEffect } from 'react';

// Create Theme Context
const ThemeContext = createContext();

// Theme Provider
export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved || 'light';
  });

  useEffect(() => {
    localStorage.setItem('theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const value = {
    theme,
    toggleTheme,
    isLight: theme === 'light',
    isDark: theme === 'dark'
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

// Custom hook
export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

// Usage in component
function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className={`header ${theme}`}>
      <h1>My App</h1>
      <button onClick={toggleTheme}>
        Switch to {theme === 'light' ? 'dark' : 'light'} mode
      </button>
    </header>
  );
}
```

### Complex Context with Authentication
```javascript
import { createContext, useContext, useReducer, useEffect } from 'react';

// Auth state and actions
const initialState = {
  user: null,
  loading: true,
  error: null
};

function authReducer(state, action) {
  switch (action.type) {
    case 'LOADING':
      return { ...state, loading: true, error: null };
    
    case 'LOGIN_SUCCESS':
      return { user: action.user, loading: false, error: null };
    
    case 'LOGIN_ERROR':
      return { user: null, loading: false, error: action.error };
    
    case 'LOGOUT':
      return { user: null, loading: false, error: null };
    
    case 'UPDATE_USER':
      return { ...state, user: { ...state.user, ...action.updates } };
    
    default:
      return state;
  }
}

// Create Auth Context
const AuthContext = createContext();

// Auth Provider
export function AuthProvider({ children }) {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Check authentication status on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          dispatch({ type: 'LOGOUT' });
          return;
        }

        const response = await fetch('/api/me', {
          headers: { Authorization: `Bearer ${token}` }
        });

        if (response.ok) {
          const user = await response.json();
          dispatch({ type: 'LOGIN_SUCCESS', user });
        } else {
          localStorage.removeItem('token');
          dispatch({ type: 'LOGOUT' });
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        dispatch({ type: 'LOGOUT' });
      }
    };

    checkAuth();
  }, []);

  // Login function
  const login = async (credentials) => {
    dispatch({ type: 'LOADING' });
    
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials)
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('token', data.token);
        dispatch({ type: 'LOGIN_SUCCESS', user: data.user });
        return data.user;
      } else {
        const error = await response.json();
        dispatch({ type: 'LOGIN_ERROR', error: error.message });
        throw new Error(error.message);
      }
    } catch (error) {
      dispatch({ type: 'LOGIN_ERROR', error: error.message });
      throw error;
    }
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem('token');
    dispatch({ type: 'LOGOUT' });
  };

  // Update user profile
  const updateUser = async (updates) => {
    try {
      const response = await fetch('/api/user', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(updates)
      });

      if (response.ok) {
        const updatedUser = await response.json();
        dispatch({ type: 'UPDATE_USER', updates: updatedUser });
        return updatedUser;
      }
    } catch (error) {
      console.error('Failed to update user:', error);
      throw error;
    }
  };

  const value = {
    ...state,
    login,
    logout,
    updateUser,
    isAuthenticated: !!state.user,
    isAdmin: state.user?.role === 'admin'
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook
export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Usage in login component
function LoginForm() {
  const [credentials, setCredentials] = useState({ email: '', password: '' });
  const { login, loading, error } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await login(credentials);
      // Redirect will be handled by route protection
    } catch (error) {
      // Error is already in context state
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={credentials.email}
        onChange={(e) => setCredentials(prev => ({ ...prev, email: e.target.value }))}
        placeholder="Email"
        required
      />
      
      <input
        type="password"
        value={credentials.password}
        onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
        placeholder="Password"
        required
      />
      
      <button type="submit" disabled={loading}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
      
      {error && <div className="error">{error}</div>}
    </form>
  );
}
```

### Multiple Context Providers Pattern
```javascript
// Combine multiple providers
function AppProviders({ children }) {
  return (
    <ThemeProvider>
      <AuthProvider>
        <NotificationProvider>
          <SettingsProvider>
            {children}
          </SettingsProvider>
        </NotificationProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

// Or create a compound provider
function createCompoundProvider(providers) {
  return ({ children }) => {
    return providers.reduceRight(
      (acc, Provider) => <Provider>{acc}</Provider>,
      children
    );
  };
}

const AppProvider = createCompoundProvider([
  ThemeProvider,
  AuthProvider,
  NotificationProvider,
  SettingsProvider
]);

// Usage
function App() {
  return (
    <AppProvider>
      <Router>
        <Routes>
          {/* Your routes */}
        </Routes>
      </Router>
    </AppProvider>
  );
}
```

## useReducer for Complex State

### Todo App with useReducer
```javascript
import { useReducer, useEffect } from 'react';

// Action types
const ACTIONS = {
  LOAD_TODOS: 'LOAD_TODOS',
  ADD_TODO: 'ADD_TODO',
  TOGGLE_TODO: 'TOGGLE_TODO',
  DELETE_TODO: 'DELETE_TODO',
  EDIT_TODO: 'EDIT_TODO',
  SET_FILTER: 'SET_FILTER',
  CLEAR_COMPLETED: 'CLEAR_COMPLETED',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR'
};

// Initial state
const initialState = {
  todos: [],
  filter: 'all', // 'all', 'active', 'completed'
  loading: false,
  error: null,
  editingId: null
};

// Reducer function
function todoReducer(state, action) {
  switch (action.type) {
    case ACTIONS.LOAD_TODOS:
      return {
        ...state,
        todos: action.todos,
        loading: false,
        error: null
      };

    case ACTIONS.ADD_TODO:
      const newTodo = {
        id: Date.now(),
        text: action.text,
        completed: false,
        createdAt: new Date().toISOString()
      };
      return {
        ...state,
        todos: [...state.todos, newTodo]
      };

    case ACTIONS.TOGGLE_TODO:
      return {
        ...state,
        todos: state.todos.map(todo =>
          todo.id === action.id
            ? { ...todo, completed: !todo.completed }
            : todo
        )
      };

    case ACTIONS.DELETE_TODO:
      return {
        ...state,
        todos: state.todos.filter(todo => todo.id !== action.id)
      };

    case ACTIONS.EDIT_TODO:
      return {
        ...state,
        todos: state.todos.map(todo =>
          todo.id === action.id
            ? { ...todo, text: action.text }
            : todo
        ),
        editingId: null
      };

    case ACTIONS.SET_FILTER:
      return {
        ...state,
        filter: action.filter
      };

    case ACTIONS.CLEAR_COMPLETED:
      return {
        ...state,
        todos: state.todos.filter(todo => !todo.completed)
      };

    case ACTIONS.SET_LOADING:
      return {
        ...state,
        loading: action.loading
      };

    case ACTIONS.SET_ERROR:
      return {
        ...state,
        error: action.error,
        loading: false
      };

    default:
      return state;
  }
}

// Custom hook for todo logic
function useTodos() {
  const [state, dispatch] = useReducer(todoReducer, initialState);

  // Load todos from API
  useEffect(() => {
    const loadTodos = async () => {
      dispatch({ type: ACTIONS.SET_LOADING, loading: true });
      
      try {
        const response = await fetch('/api/todos');
        const todos = await response.json();
        dispatch({ type: ACTIONS.LOAD_TODOS, todos });
      } catch (error) {
        dispatch({ type: ACTIONS.SET_ERROR, error: error.message });
      }
    };

    loadTodos();
  }, []);

  // Action creators
  const addTodo = (text) => {
    if (text.trim()) {
      dispatch({ type: ACTIONS.ADD_TODO, text: text.trim() });
    }
  };

  const toggleTodo = (id) => {
    dispatch({ type: ACTIONS.TOGGLE_TODO, id });
  };

  const deleteTodo = (id) => {
    dispatch({ type: ACTIONS.DELETE_TODO, id });
  };

  const editTodo = (id, text) => {
    if (text.trim()) {
      dispatch({ type: ACTIONS.EDIT_TODO, id, text: text.trim() });
    }
  };

  const setFilter = (filter) => {
    dispatch({ type: ACTIONS.SET_FILTER, filter });
  };

  const clearCompleted = () => {
    dispatch({ type: ACTIONS.CLEAR_COMPLETED });
  };

  // Selectors
  const filteredTodos = state.todos.filter(todo => {
    switch (state.filter) {
      case 'active':
        return !todo.completed;
      case 'completed':
        return todo.completed;
      default:
        return true;
    }
  });

  const activeCount = state.todos.filter(todo => !todo.completed).length;
  const completedCount = state.todos.filter(todo => todo.completed).length;

  return {
    ...state,
    filteredTodos,
    activeCount,
    completedCount,
    addTodo,
    toggleTodo,
    deleteTodo,
    editTodo,
    setFilter,
    clearCompleted
  };
}

// Todo App component
function TodoApp() {
  const {
    filteredTodos,
    filter,
    activeCount,
    completedCount,
    loading,
    error,
    addTodo,
    toggleTodo,
    deleteTodo,
    editTodo,
    setFilter,
    clearCompleted
  } = useTodos();

  const [newTodoText, setNewTodoText] = useState('');

  const handleAddTodo = (e) => {
    e.preventDefault();
    addTodo(newTodoText);
    setNewTodoText('');
  };

  if (loading) return <div>Loading todos...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="todo-app">
      <h1>Todo App</h1>
      
      <form onSubmit={handleAddTodo}>
        <input
          type="text"
          value={newTodoText}
          onChange={(e) => setNewTodoText(e.target.value)}
          placeholder="Add a new todo..."
        />
        <button type="submit">Add</button>
      </form>

      <div className="todo-filters">
        <button 
          className={filter === 'all' ? 'active' : ''}
          onClick={() => setFilter('all')}
        >
          All ({activeCount + completedCount})
        </button>
        <button 
          className={filter === 'active' ? 'active' : ''}
          onClick={() => setFilter('active')}
        >
          Active ({activeCount})
        </button>
        <button 
          className={filter === 'completed' ? 'active' : ''}
          onClick={() => setFilter('completed')}
        >
          Completed ({completedCount})
        </button>
      </div>

      <ul className="todo-list">
        {filteredTodos.map(todo => (
          <TodoItem
            key={todo.id}
            todo={todo}
            onToggle={() => toggleTodo(todo.id)}
            onDelete={() => deleteTodo(todo.id)}
            onEdit={(text) => editTodo(todo.id, text)}
          />
        ))}
      </ul>

      {completedCount > 0 && (
        <button onClick={clearCompleted}>
          Clear Completed ({completedCount})
        </button>
      )}
    </div>
  );
}

// Todo Item component
function TodoItem({ todo, onToggle, onDelete, onEdit }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(todo.text);

  const handleEdit = () => {
    onEdit(editText);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditText(todo.text);
    setIsEditing(false);
  };

  return (
    <li className={`todo-item ${todo.completed ? 'completed' : ''}`}>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={onToggle}
      />
      
      {isEditing ? (
        <div className="edit-form">
          <input
            type="text"
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleEdit()}
            autoFocus
          />
          <button onClick={handleEdit}>Save</button>
          <button onClick={handleCancel}>Cancel</button>
        </div>
      ) : (
        <div className="todo-content">
          <span onClick={() => setIsEditing(true)}>{todo.text}</span>
          <button onClick={() => setIsEditing(true)}>Edit</button>
          <button onClick={onDelete}>Delete</button>
        </div>
      )}
    </li>
  );
}
```

## Redux Toolkit

### Store Setup
```javascript
// store/store.js
import { configureStore } from '@reduxjs/toolkit';
import authSlice from './authSlice';
import todoSlice from './todoSlice';
import uiSlice from './uiSlice';

export const store = configureStore({
  reducer: {
    auth: authSlice,
    todos: todoSlice,
    ui: uiSlice
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST']
      }
    }),
  devTools: process.env.NODE_ENV !== 'production'
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

### Creating Slices
```javascript
// store/authSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// Async thunks
export const loginUser = createAsyncThunk(
  'auth/loginUser',
  async (credentials, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials)
      });

      if (!response.ok) {
        const error = await response.json();
        return rejectWithValue(error.message);
      }

      const data = await response.json();
      localStorage.setItem('token', data.token);
      return data.user;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const logoutUser = createAsyncThunk(
  'auth/logoutUser',
  async () => {
    localStorage.removeItem('token');
    return null;
  }
);

export const checkAuthStatus = createAsyncThunk(
  'auth/checkAuthStatus',
  async (_, { rejectWithValue }) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return null;

      const response = await fetch('/api/me', {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (!response.ok) {
        localStorage.removeItem('token');
        return null;
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

// Slice
const authSlice = createSlice({
  name: 'auth',
  initialState: {
    user: null,
    loading: false,
    error: null,
    isAuthenticated: false
  },
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    updateUser: (state, action) => {
      if (state.user) {
        state.user = { ...state.user, ...action.payload };
      }
    }
  },
  extraReducers: (builder) => {
    builder
      // Login
      .addCase(loginUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
        state.isAuthenticated = true;
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
        state.isAuthenticated = false;
      })
      
      // Logout
      .addCase(logoutUser.fulfilled, (state) => {
        state.user = null;
        state.isAuthenticated = false;
      })
      
      // Check auth status
      .addCase(checkAuthStatus.pending, (state) => {
        state.loading = true;
      })
      .addCase(checkAuthStatus.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
        state.isAuthenticated = !!action.payload;
      })
      .addCase(checkAuthStatus.rejected, (state) => {
        state.loading = false;
        state.isAuthenticated = false;
      });
  }
});

export const { clearError, updateUser } = authSlice.actions;
export default authSlice.reducer;
```

### Todo Slice with RTK Query
```javascript
// store/todoSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// Async thunks
export const fetchTodos = createAsyncThunk(
  'todos/fetchTodos',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/todos');
      if (!response.ok) throw new Error('Failed to fetch todos');
      return await response.json();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const createTodo = createAsyncThunk(
  'todos/createTodo',
  async (todoData, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/todos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(todoData)
      });
      if (!response.ok) throw new Error('Failed to create todo');
      return await response.json();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const updateTodo = createAsyncThunk(
  'todos/updateTodo',
  async ({ id, updates }, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/todos/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });
      if (!response.ok) throw new Error('Failed to update todo');
      return await response.json();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const deleteTodo = createAsyncThunk(
  'todos/deleteTodo',
  async (id, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/todos/${id}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to delete todo');
      return id;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const todoSlice = createSlice({
  name: 'todos',
  initialState: {
    items: [],
    filter: 'all',
    loading: false,
    error: null
  },
  reducers: {
    setFilter: (state, action) => {
      state.filter = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    }
  },
  extraReducers: (builder) => {
    builder
      // Fetch todos
      .addCase(fetchTodos.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTodos.fulfilled, (state, action) => {
        state.loading = false;
        state.items = action.payload;
      })
      .addCase(fetchTodos.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      
      // Create todo
      .addCase(createTodo.fulfilled, (state, action) => {
        state.items.push(action.payload);
      })
      
      // Update todo
      .addCase(updateTodo.fulfilled, (state, action) => {
        const index = state.items.findIndex(todo => todo.id === action.payload.id);
        if (index !== -1) {
          state.items[index] = action.payload;
        }
      })
      
      // Delete todo
      .addCase(deleteTodo.fulfilled, (state, action) => {
        state.items = state.items.filter(todo => todo.id !== action.payload);
      });
  }
});

export const { setFilter, clearError } = todoSlice.actions;
export default todoSlice.reducer;
```

### Using Redux in Components
```javascript
// hooks/redux.js
import { useDispatch, useSelector, TypedUseSelectorHook } from 'react-redux';
import type { RootState, AppDispatch } from '../store/store';

export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

// components/TodoApp.jsx
import { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { 
  fetchTodos, 
  createTodo, 
  updateTodo, 
  deleteTodo, 
  setFilter 
} from '../store/todoSlice';

function TodoApp() {
  const dispatch = useAppDispatch();
  const { items: todos, filter, loading, error } = useAppSelector(state => state.todos);
  
  useEffect(() => {
    dispatch(fetchTodos());
  }, [dispatch]);

  const handleAddTodo = (text) => {
    dispatch(createTodo({ text, completed: false }));
  };

  const handleToggleTodo = (id) => {
    const todo = todos.find(t => t.id === id);
    if (todo) {
      dispatch(updateTodo({ id, updates: { completed: !todo.completed } }));
    }
  };

  const handleDeleteTodo = (id) => {
    dispatch(deleteTodo(id));
  };

  const filteredTodos = todos.filter(todo => {
    switch (filter) {
      case 'active': return !todo.completed;
      case 'completed': return todo.completed;
      default: return true;
    }
  });

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h1>Todo App</h1>
      
      <TodoForm onSubmit={handleAddTodo} />
      
      <div>
        <button 
          className={filter === 'all' ? 'active' : ''}
          onClick={() => dispatch(setFilter('all'))}
        >
          All
        </button>
        <button 
          className={filter === 'active' ? 'active' : ''}
          onClick={() => dispatch(setFilter('active'))}
        >
          Active
        </button>
        <button 
          className={filter === 'completed' ? 'active' : ''}
          onClick={() => dispatch(setFilter('completed'))}
        >
          Completed
        </button>
      </div>

      <ul>
        {filteredTodos.map(todo => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => handleToggleTodo(todo.id)}
            />
            <span>{todo.text}</span>
            <button onClick={() => handleDeleteTodo(todo.id)}>
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

## Zustand

### Basic Store Setup
```javascript
// stores/useStore.js
import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';

// Simple counter store
export const useCounterStore = create(
  devtools((set, get) => ({
    count: 0,
    increment: () => set((state) => ({ count: state.count + 1 })),
    decrement: () => set((state) => ({ count: state.count - 1 })),
    reset: () => set({ count: 0 }),
    incrementBy: (amount) => set((state) => ({ count: state.count + amount }))
  }))
);

// Todo store with persistence
export const useTodoStore = create(
  devtools(
    persist(
      (set, get) => ({
        todos: [],
        filter: 'all',
        
        // Actions
        addTodo: (text) => {
          const newTodo = {
            id: Date.now(),
            text,
            completed: false,
            createdAt: new Date().toISOString()
          };
          set((state) => ({ todos: [...state.todos, newTodo] }));
        },
        
        toggleTodo: (id) => {
          set((state) => ({
            todos: state.todos.map(todo =>
              todo.id === id ? { ...todo, completed: !todo.completed } : todo
            )
          }));
        },
        
        deleteTodo: (id) => {
          set((state) => ({
            todos: state.todos.filter(todo => todo.id !== id)
          }));
        },
        
        editTodo: (id, text) => {
          set((state) => ({
            todos: state.todos.map(todo =>
              todo.id === id ? { ...todo, text } : todo
            )
          }));
        },
        
        setFilter: (filter) => set({ filter }),
        
        clearCompleted: () => {
          set((state) => ({
            todos: state.todos.filter(todo => !todo.completed)
          }));
        },
        
        // Selectors
        get filteredTodos() {
          const { todos, filter } = get();
          switch (filter) {
            case 'active':
              return todos.filter(todo => !todo.completed);
            case 'completed':
              return todos.filter(todo => todo.completed);
            default:
              return todos;
          }
        },
        
        get activeCount() {
          return get().todos.filter(todo => !todo.completed).length;
        },
        
        get completedCount() {
          return get().todos.filter(todo => todo.completed).length;
        }
      }),
      {
        name: 'todo-storage', // name of the localStorage key
        getStorage: () => localStorage
      }
    )
  )
);

// Auth store with async actions
export const useAuthStore = create(
  devtools((set, get) => ({
    user: null,
    loading: false,
    error: null,
    
    login: async (credentials) => {
      set({ loading: true, error: null });
      
      try {
        const response = await fetch('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(credentials)
        });
        
        if (!response.ok) {
          throw new Error('Login failed');
        }
        
        const data = await response.json();
        localStorage.setItem('token', data.token);
        set({ user: data.user, loading: false });
        
        return data.user;
      } catch (error) {
        set({ error: error.message, loading: false });
        throw error;
      }
    },
    
    logout: () => {
      localStorage.removeItem('token');
      set({ user: null, error: null });
    },
    
    checkAuth: async () => {
      const token = localStorage.getItem('token');
      if (!token) return;
      
      set({ loading: true });
      
      try {
        const response = await fetch('/api/me', {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        if (response.ok) {
          const user = await response.json();
          set({ user, loading: false });
        } else {
          localStorage.removeItem('token');
          set({ user: null, loading: false });
        }
      } catch (error) {
        set({ error: error.message, loading: false });
      }
    },
    
    updateUser: (updates) => {
      set((state) => ({
        user: state.user ? { ...state.user, ...updates } : null
      }));
    },
    
    clearError: () => set({ error: null })
  }))
);
```

### Using Zustand in Components
```javascript
// components/Counter.jsx
import { useCounterStore } from '../stores/useStore';

function Counter() {
  const { count, increment, decrement, reset, incrementBy } = useCounterStore();

  return (
    <div>
      <h2>Count: {count}</h2>
      <button onClick={increment}>+1</button>
      <button onClick={decrement}>-1</button>
      <button onClick={() => incrementBy(5)}>+5</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}

// components/TodoApp.jsx
import { useTodoStore } from '../stores/useStore';

function TodoApp() {
  const {
    filteredTodos,
    filter,
    activeCount,
    completedCount,
    addTodo,
    toggleTodo,
    deleteTodo,
    setFilter,
    clearCompleted
  } = useTodoStore();

  const [newTodoText, setNewTodoText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (newTodoText.trim()) {
      addTodo(newTodoText.trim());
      setNewTodoText('');
    }
  };

  return (
    <div>
      <h1>Todo App</h1>
      
      <form onSubmit={handleSubmit}>
        <input
          value={newTodoText}
          onChange={(e) => setNewTodoText(e.target.value)}
          placeholder="Add todo..."
        />
        <button type="submit">Add</button>
      </form>

      <div>
        <button onClick={() => setFilter('all')}>
          All ({activeCount + completedCount})
        </button>
        <button onClick={() => setFilter('active')}>
          Active ({activeCount})
        </button>
        <button onClick={() => setFilter('completed')}>
          Completed ({completedCount})
        </button>
      </div>

      <ul>
        {filteredTodos.map(todo => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleTodo(todo.id)}
            />
            <span>{todo.text}</span>
            <button onClick={() => deleteTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>

      {completedCount > 0 && (
        <button onClick={clearCompleted}>
          Clear Completed ({completedCount})
        </button>
      )}
    </div>
  );
}

// Using specific state slices to optimize re-renders
function TodoStats() {
  // Only subscribe to the specific values we need
  const activeCount = useTodoStore(state => state.activeCount);
  const completedCount = useTodoStore(state => state.completedCount);
  
  return (
    <div>
      <p>Active: {activeCount}</p>
      <p>Completed: {completedCount}</p>
    </div>
  );
}
```

### Zustand with TypeScript
```typescript
// stores/types.ts
export interface Todo {
  id: number;
  text: string;
  completed: boolean;
  createdAt: string;
}

export interface User {
  id: number;
  name: string;
  email: string;
  role: 'user' | 'admin';
}

// stores/todoStore.ts
import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';
import type { Todo } from './types';

interface TodoState {
  todos: Todo[];
  filter: 'all' | 'active' | 'completed';
  addTodo: (text: string) => void;
  toggleTodo: (id: number) => void;
  deleteTodo: (id: number) => void;
  editTodo: (id: number, text: string) => void;
  setFilter: (filter: 'all' | 'active' | 'completed') => void;
  clearCompleted: () => void;
  filteredTodos: Todo[];
  activeCount: number;
  completedCount: number;
}

export const useTodoStore = create<TodoState>()(
  devtools(
    persist(
      (set, get) => ({
        todos: [],
        filter: 'all',
        
        addTodo: (text: string) => {
          const newTodo: Todo = {
            id: Date.now(),
            text,
            completed: false,
            createdAt: new Date().toISOString()
          };
          set((state) => ({ todos: [...state.todos, newTodo] }));
        },
        
        toggleTodo: (id: number) => {
          set((state) => ({
            todos: state.todos.map(todo =>
              todo.id === id ? { ...todo, completed: !todo.completed } : todo
            )
          }));
        },
        
        deleteTodo: (id: number) => {
          set((state) => ({
            todos: state.todos.filter(todo => todo.id !== id)
          }));
        },
        
        editTodo: (id: number, text: string) => {
          set((state) => ({
            todos: state.todos.map(todo =>
              todo.id === id ? { ...todo, text } : todo
            )
          }));
        },
        
        setFilter: (filter) => set({ filter }),
        
        clearCompleted: () => {
          set((state) => ({
            todos: state.todos.filter(todo => !todo.completed)
          }));
        },
        
        get filteredTodos() {
          const { todos, filter } = get();
          switch (filter) {
            case 'active':
              return todos.filter(todo => !todo.completed);
            case 'completed':
              return todos.filter(todo => todo.completed);
            default:
              return todos;
          }
        },
        
        get activeCount() {
          return get().todos.filter(todo => !todo.completed).length;
        },
        
        get completedCount() {
          return get().todos.filter(todo => todo.completed).length;
        }
      }),
      {
        name: 'todo-storage'
      }
    )
  )
);
```

## State Management Patterns

### Lifting State Up
```javascript
// When multiple components need the same state
function App() {
  const [selectedUser, setSelectedUser] = useState(null);
  const [users, setUsers] = useState([]);

  return (
    <div>
      <UserList 
        users={users}
        selectedUser={selectedUser}
        onSelectUser={setSelectedUser}
      />
      <UserDetails 
        user={selectedUser}
        onUpdateUser={(updatedUser) => {
          setUsers(prev => prev.map(user => 
            user.id === updatedUser.id ? updatedUser : user
          ));
          setSelectedUser(updatedUser);
        }}
      />
    </div>
  );
}
```

### Compound Components with State
```javascript
function Accordion({ children, allowMultiple = false }) {
  const [openItems, setOpenItems] = useState(new Set());
  
  const toggleItem = (id) => {
    setOpenItems(prev => {
      const newSet = new Set(prev);
      
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        if (!allowMultiple) {
          newSet.clear();
        }
        newSet.add(id);
      }
      
      return newSet;
    });
  };

  return (
    <div className="accordion">
      {React.Children.map(children, (child, index) =>
        React.cloneElement(child, {
          isOpen: openItems.has(index),
          onToggle: () => toggleItem(index),
          id: index
        })
      )}
    </div>
  );
}

function AccordionItem({ children, title, isOpen, onToggle }) {
  return (
    <div className="accordion-item">
      <button 
        className="accordion-header"
        onClick={onToggle}
      >
        {title}
      </button>
      {isOpen && (
        <div className="accordion-content">
          {children}
        </div>
      )}
    </div>
  );
}

// Usage
<Accordion allowMultiple>
  <AccordionItem title="Section 1">
    Content for section 1
  </AccordionItem>
  <AccordionItem title="Section 2">
    Content for section 2
  </AccordionItem>
</Accordion>
```

## Best Practices

### State Management Guidelines

1. **Start Simple**: Begin with `useState` and lift state up as needed
2. **Use Context Sparingly**: Only for truly global state (theme, auth)
3. **Avoid Context for Frequently Changing State**: Can cause performance issues
4. **Consider useReducer**: For complex state logic with multiple actions
5. **Choose External Libraries Wisely**: Redux for complex apps, Zustand for simplicity

### Performance Optimization
```javascript
// Minimize re-renders with proper state structure
// ❌ Bad - causes unnecessary re-renders
const [state, setState] = useState({
  user: null,
  posts: [],
  ui: { loading: false, error: null }
});

// ✅ Good - separate concerns
const [user, setUser] = useState(null);
const [posts, setPosts] = useState([]);
const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);

// ❌ Bad - creating new objects in render
function UserProfile({ user }) {
  const userStyle = { color: user.isActive ? 'green' : 'red' }; // New object every render
  return <div style={userStyle}>{user.name}</div>;
}

// ✅ Good - memoize or move outside component
const activeStyle = { color: 'green' };
const inactiveStyle = { color: 'red' };

function UserProfile({ user }) {
  const userStyle = user.isActive ? activeStyle : inactiveStyle;
  return <div style={userStyle}>{user.name}</div>;
}

// ✅ Better - use CSS classes
function UserProfile({ user }) {
  return (
    <div className={`user ${user.isActive ? 'active' : 'inactive'}`}>
      {user.name}
    </div>
  );
}
```

### Testing State Management
```javascript
// Testing custom hooks
import { renderHook, act } from '@testing-library/react';
import { useTodos } from './useTodos';

test('should add a todo', () => {
  const { result } = renderHook(() => useTodos());
  
  act(() => {
    result.current.addTodo('Test todo');
  });
  
  expect(result.current.todos).toHaveLength(1);
  expect(result.current.todos[0].text).toBe('Test todo');
});

// Testing Redux
import { store } from './store';
import { addTodo } from './todoSlice';

test('should add todo to store', () => {
  const initialState = store.getState();
  expect(initialState.todos.items).toHaveLength(0);
  
  store.dispatch(addTodo({ text: 'Test todo' }));
  
  const newState = store.getState();
  expect(newState.todos.items).toHaveLength(1);
});

// Testing Zustand
import { useTodoStore } from './todoStore';

test('should add todo to Zustand store', () => {
  const { addTodo, todos } = useTodoStore.getState();
  
  expect(todos).toHaveLength(0);
  
  addTodo('Test todo');
  
  expect(useTodoStore.getState().todos).toHaveLength(1);
});
```

---

*Continue to: [06-react-performance.md](./06-react-performance.md)*
