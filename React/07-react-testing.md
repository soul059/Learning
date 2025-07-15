# React Testing

## Table of Contents
- [Testing Overview](#testing-overview)
- [Jest Fundamentals](#jest-fundamentals)
- [React Testing Library](#react-testing-library)
- [Component Testing](#component-testing)
- [Hook Testing](#hook-testing)
- [Integration Testing](#integration-testing)
- [Mocking](#mocking)
- [E2E Testing](#e2e-testing)
- [Testing Best Practices](#testing-best-practices)

## Testing Overview

### Types of Testing
```javascript
// Testing Pyramid
// E2E Tests (Few, Slow, High Confidence)
//   ↑
// Integration Tests (Some, Medium Speed)
//   ↑  
// Unit Tests (Many, Fast, Low Level)

// Testing Philosophy in React
// 1. Test behavior, not implementation
// 2. Test what users see and do
// 3. Avoid testing internal state
// 4. Use realistic test data
// 5. Keep tests simple and focused
```

### Setting Up Testing Environment
```javascript
// package.json
{
  "dependencies": {
    "@testing-library/react": "^13.0.0",
    "@testing-library/jest-dom": "^5.16.0",
    "@testing-library/user-event": "^14.0.0"
  },
  "scripts": {
    "test": "react-scripts test",
    "test:coverage": "react-scripts test --coverage --watchAll=false",
    "test:debug": "react-scripts --inspect-brk test --runInBand --no-cache"
  }
}

// setupTests.js
import '@testing-library/jest-dom';

// Custom matchers
expect.extend({
  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

// jest.config.js
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.js'],
  moduleNameMapping: {
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|eot|otf|webp|svg|ttf|woff|woff2)$': '<rootDir>/__mocks__/fileMock.js'
  },
  collectCoverageFrom: [
    'src/**/*.{js,jsx}',
    '!src/index.js',
    '!src/reportWebVitals.js'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

## Jest Fundamentals

### Basic Test Structure
```javascript
// Math.test.js
describe('Math utilities', () => {
  beforeAll(() => {
    console.log('Setup before all tests');
  });

  beforeEach(() => {
    console.log('Setup before each test');
  });

  afterEach(() => {
    console.log('Cleanup after each test');
  });

  afterAll(() => {
    console.log('Cleanup after all tests');
  });

  describe('addition', () => {
    test('adds 1 + 2 to equal 3', () => {
      expect(1 + 2).toBe(3);
    });

    test('adds positive numbers', () => {
      expect(2 + 3).toBe(5);
      expect(10 + 5).toBe(15);
    });

    test.each([
      [1, 1, 2],
      [1, 2, 3],
      [2, 1, 3],
    ])('add(%i, %i) = %i', (a, b, expected) => {
      expect(a + b).toBe(expected);
    });
  });

  describe('subtraction', () => {
    test('subtracts numbers correctly', () => {
      expect(5 - 3).toBe(2);
    });

    test.skip('skip this test for now', () => {
      expect(true).toBe(false);
    });

    test.only('only run this test', () => {
      expect(true).toBe(true);
    });
  });
});
```

### Jest Matchers
```javascript
describe('Jest matchers', () => {
  test('common matchers', () => {
    // Equality
    expect(2 + 2).toBe(4);
    expect({ name: 'John' }).toEqual({ name: 'John' });
    expect({ name: 'John' }).not.toBe({ name: 'John' }); // Different objects

    // Truthiness
    expect(true).toBeTruthy();
    expect(false).toBeFalsy();
    expect(null).toBeNull();
    expect(undefined).toBeUndefined();
    expect('Hello').toBeDefined();

    // Numbers
    expect(2 + 2).toBeGreaterThan(3);
    expect(Math.PI).toBeCloseTo(3.14159, 5);

    // Strings
    expect('team').toMatch(/I/);
    expect('Christoph').toMatch('stop');

    // Arrays
    expect(['Alice', 'Bob', 'Eve']).toContain('Alice');
    expect(['a', 'b', 'c']).toHaveLength(3);

    // Objects
    expect({ name: 'John', age: 30 }).toHaveProperty('name');
    expect({ name: 'John', age: 30 }).toHaveProperty('name', 'John');

    // Exceptions
    expect(() => {
      throw new Error('Something went wrong');
    }).toThrow('Something went wrong');
  });

  test('async matchers', async () => {
    // Promises
    await expect(Promise.resolve('success')).resolves.toBe('success');
    await expect(Promise.reject('error')).rejects.toBe('error');

    // Async function
    const fetchData = async () => 'data';
    await expect(fetchData()).resolves.toBe('data');
  });
});
```

### Mocking with Jest
```javascript
// Simple mocks
describe('Mocking', () => {
  test('mock function', () => {
    const mockFn = jest.fn();
    mockFn('arg1', 'arg2');
    mockFn('arg3');

    expect(mockFn).toHaveBeenCalledTimes(2);
    expect(mockFn).toHaveBeenCalledWith('arg1', 'arg2');
    expect(mockFn).toHaveBeenLastCalledWith('arg3');
    expect(mockFn).toHaveBeenNthCalledWith(1, 'arg1', 'arg2');
  });

  test('mock return value', () => {
    const mockFn = jest.fn();
    mockFn.mockReturnValue(42);
    mockFn.mockReturnValueOnce(10);

    expect(mockFn()).toBe(10); // First call
    expect(mockFn()).toBe(42); // Subsequent calls
  });

  test('mock implementation', () => {
    const mockFn = jest.fn((x) => x * 2);
    expect(mockFn(5)).toBe(10);

    mockFn.mockImplementation((x) => x * 3);
    expect(mockFn(5)).toBe(15);
  });

  test('mock modules', () => {
    // Mock entire module
    jest.mock('./math', () => ({
      add: jest.fn(() => 10),
      subtract: jest.fn(() => 5),
    }));

    const math = require('./math');
    expect(math.add()).toBe(10);
  });
});

// __mocks__/axios.js
export default {
  get: jest.fn(() => Promise.resolve({ data: {} })),
  post: jest.fn(() => Promise.resolve({ data: {} })),
  put: jest.fn(() => Promise.resolve({ data: {} })),
  delete: jest.fn(() => Promise.resolve({ data: {} })),
};
```

## React Testing Library

### Basic Component Testing
```javascript
import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Button from './Button';

// Button.js
function Button({ onClick, children, disabled = false, variant = 'primary' }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`btn btn-${variant}`}
      data-testid="button"
    >
      {children}
    </button>
  );
}

// Button.test.js
describe('Button Component', () => {
  test('renders button with text', () => {
    render(<Button>Click me</Button>);
    
    // Query by text
    expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument();
    
    // Query by test id
    expect(screen.getByTestId('button')).toBeInTheDocument();
  });

  test('calls onClick when clicked', async () => {
    const user = userEvent.setup();
    const handleClick = jest.fn();
    
    render(<Button onClick={handleClick}>Click me</Button>);
    
    await user.click(screen.getByRole('button'));
    
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  test('is disabled when disabled prop is true', () => {
    render(<Button disabled>Disabled</Button>);
    
    expect(screen.getByRole('button')).toBeDisabled();
  });

  test('applies correct CSS class for variant', () => {
    render(<Button variant="secondary">Secondary</Button>);
    
    expect(screen.getByRole('button')).toHaveClass('btn-secondary');
  });

  test('does not call onClick when disabled', async () => {
    const user = userEvent.setup();
    const handleClick = jest.fn();
    
    render(<Button onClick={handleClick} disabled>Disabled</Button>);
    
    await user.click(screen.getByRole('button'));
    
    expect(handleClick).not.toHaveBeenCalled();
  });
});
```

### Form Testing
```javascript
import React, { useState } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// LoginForm.js
function LoginForm({ onSubmit }) {
  const [formData, setFormData] = useState({ email: '', password: '' });
  const [errors, setErrors] = useState({});

  const handleSubmit = (e) => {
    e.preventDefault();
    const newErrors = {};

    if (!formData.email) newErrors.email = 'Email is required';
    if (!formData.password) newErrors.password = 'Password is required';

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label htmlFor="email">Email:</label>
        <input
          id="email"
          type="email"
          value={formData.email}
          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
        />
        {errors.email && <span role="alert">{errors.email}</span>}
      </div>

      <div>
        <label htmlFor="password">Password:</label>
        <input
          id="password"
          type="password"
          value={formData.password}
          onChange={(e) => setFormData({ ...formData, password: e.target.value })}
        />
        {errors.password && <span role="alert">{errors.password}</span>}
      </div>

      <button type="submit">Login</button>
    </form>
  );
}

// LoginForm.test.js
describe('LoginForm', () => {
  test('submits form with valid data', async () => {
    const user = userEvent.setup();
    const mockSubmit = jest.fn();

    render(<LoginForm onSubmit={mockSubmit} />);

    // Fill out form
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/password/i), 'password123');

    // Submit form
    await user.click(screen.getByRole('button', { name: /login/i }));

    expect(mockSubmit).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'password123'
    });
  });

  test('shows validation errors for empty fields', async () => {
    const user = userEvent.setup();
    const mockSubmit = jest.fn();

    render(<LoginForm onSubmit={mockSubmit} />);

    // Submit empty form
    await user.click(screen.getByRole('button', { name: /login/i }));

    expect(screen.getByText('Email is required')).toBeInTheDocument();
    expect(screen.getByText('Password is required')).toBeInTheDocument();
    expect(mockSubmit).not.toHaveBeenCalled();
  });

  test('clears error when user types in field', async () => {
    const user = userEvent.setup();

    render(<LoginForm onSubmit={jest.fn()} />);

    // Submit to trigger errors
    await user.click(screen.getByRole('button', { name: /login/i }));
    
    expect(screen.getByText('Email is required')).toBeInTheDocument();

    // Type in email field
    await user.type(screen.getByLabelText(/email/i), 'test@example.com');

    // Error should be cleared (this would need additional logic in the component)
  });
});
```

### Testing with Context
```javascript
import React, { createContext, useContext, useState } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Theme context
const ThemeContext = createContext();

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');
  
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useContext(ThemeContext);
  
  return (
    <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
      Current theme: {theme}
    </button>
  );
}

// Test utility for rendering with providers
function renderWithTheme(ui, options = {}) {
  function Wrapper({ children }) {
    return <ThemeProvider>{children}</ThemeProvider>;
  }
  
  return render(ui, { wrapper: Wrapper, ...options });
}

// ThemeToggle.test.js
describe('ThemeToggle', () => {
  test('displays current theme', () => {
    renderWithTheme(<ThemeToggle />);
    
    expect(screen.getByText(/current theme: light/i)).toBeInTheDocument();
  });

  test('toggles theme when clicked', async () => {
    const user = userEvent.setup();
    
    renderWithTheme(<ThemeToggle />);
    
    const button = screen.getByRole('button');
    
    expect(button).toHaveTextContent('Current theme: light');
    
    await user.click(button);
    
    expect(button).toHaveTextContent('Current theme: dark');
    
    await user.click(button);
    
    expect(button).toHaveTextContent('Current theme: light');
  });

  test('works with custom initial theme', () => {
    function CustomThemeProvider({ children }) {
      const [theme, setTheme] = useState('dark');
      return (
        <ThemeContext.Provider value={{ theme, setTheme }}>
          {children}
        </ThemeContext.Provider>
      );
    }

    render(
      <CustomThemeProvider>
        <ThemeToggle />
      </CustomThemeProvider>
    );

    expect(screen.getByText(/current theme: dark/i)).toBeInTheDocument();
  });
});
```

## Component Testing

### Testing Props and State
```javascript
import React, { useState } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Counter.js
function Counter({ initialCount = 0, max = 10, onCountChange }) {
  const [count, setCount] = useState(initialCount);

  const increment = () => {
    if (count < max) {
      const newCount = count + 1;
      setCount(newCount);
      onCountChange?.(newCount);
    }
  };

  const decrement = () => {
    if (count > 0) {
      const newCount = count - 1;
      setCount(newCount);
      onCountChange?.(newCount);
    }
  };

  return (
    <div>
      <span data-testid="count">Count: {count}</span>
      <button onClick={increment} disabled={count >= max}>
        +
      </button>
      <button onClick={decrement} disabled={count <= 0}>
        -
      </button>
    </div>
  );
}

// Counter.test.js
describe('Counter', () => {
  test('renders with initial count', () => {
    render(<Counter initialCount={5} />);
    
    expect(screen.getByTestId('count')).toHaveTextContent('Count: 5');
  });

  test('increments count when + button is clicked', async () => {
    const user = userEvent.setup();
    
    render(<Counter />);
    
    await user.click(screen.getByText('+'));
    
    expect(screen.getByTestId('count')).toHaveTextContent('Count: 1');
  });

  test('decrements count when - button is clicked', async () => {
    const user = userEvent.setup();
    
    render(<Counter initialCount={1} />);
    
    await user.click(screen.getByText('-'));
    
    expect(screen.getByTestId('count')).toHaveTextContent('Count: 0');
  });

  test('disables + button when max is reached', async () => {
    const user = userEvent.setup();
    
    render(<Counter initialCount={2} max={3} />);
    
    await user.click(screen.getByText('+')); // count = 3
    
    expect(screen.getByText('+')).toBeDisabled();
  });

  test('disables - button when count is 0', () => {
    render(<Counter initialCount={0} />);
    
    expect(screen.getByText('-')).toBeDisabled();
  });

  test('calls onCountChange when count changes', async () => {
    const user = userEvent.setup();
    const mockOnCountChange = jest.fn();
    
    render(<Counter onCountChange={mockOnCountChange} />);
    
    await user.click(screen.getByText('+'));
    
    expect(mockOnCountChange).toHaveBeenCalledWith(1);
  });
});
```

### Testing Conditional Rendering
```javascript
import React from 'react';
import { render, screen } from '@testing-library/react';

// UserProfile.js
function UserProfile({ user, isLoading, error }) {
  if (isLoading) {
    return <div data-testid="loading">Loading...</div>;
  }

  if (error) {
    return <div data-testid="error">Error: {error}</div>;
  }

  if (!user) {
    return <div data-testid="no-user">No user found</div>;
  }

  return (
    <div data-testid="user-profile">
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      {user.avatar && <img src={user.avatar} alt="User avatar" />}
    </div>
  );
}

// UserProfile.test.js
describe('UserProfile', () => {
  const mockUser = {
    name: 'John Doe',
    email: 'john@example.com',
    avatar: 'https://example.com/avatar.jpg'
  };

  test('renders loading state', () => {
    render(<UserProfile isLoading={true} />);
    
    expect(screen.getByTestId('loading')).toBeInTheDocument();
    expect(screen.queryByTestId('user-profile')).not.toBeInTheDocument();
  });

  test('renders error state', () => {
    render(<UserProfile error="Failed to load user" />);
    
    expect(screen.getByTestId('error')).toHaveTextContent('Error: Failed to load user');
  });

  test('renders no user state', () => {
    render(<UserProfile user={null} />);
    
    expect(screen.getByTestId('no-user')).toBeInTheDocument();
  });

  test('renders user profile with all data', () => {
    render(<UserProfile user={mockUser} />);
    
    expect(screen.getByTestId('user-profile')).toBeInTheDocument();
    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('john@example.com')).toBeInTheDocument();
    expect(screen.getByAltText('User avatar')).toHaveAttribute('src', mockUser.avatar);
  });

  test('renders user profile without avatar', () => {
    const userWithoutAvatar = { ...mockUser, avatar: null };
    
    render(<UserProfile user={userWithoutAvatar} />);
    
    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.queryByAltText('User avatar')).not.toBeInTheDocument();
  });
});
```

## Hook Testing

### Testing Custom Hooks
```javascript
import { renderHook, act } from '@testing-library/react';
import { useState, useEffect } from 'react';

// useCounter.js
function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);

  const increment = () => setCount(prev => prev + 1);
  const decrement = () => setCount(prev => prev - 1);
  const reset = () => setCount(initialValue);

  return { count, increment, decrement, reset };
}

// useCounter.test.js
describe('useCounter', () => {
  test('initializes with default value', () => {
    const { result } = renderHook(() => useCounter());
    
    expect(result.current.count).toBe(0);
  });

  test('initializes with custom value', () => {
    const { result } = renderHook(() => useCounter(10));
    
    expect(result.current.count).toBe(10);
  });

  test('increments count', () => {
    const { result } = renderHook(() => useCounter());
    
    act(() => {
      result.current.increment();
    });
    
    expect(result.current.count).toBe(1);
  });

  test('decrements count', () => {
    const { result } = renderHook(() => useCounter(5));
    
    act(() => {
      result.current.decrement();
    });
    
    expect(result.current.count).toBe(4);
  });

  test('resets to initial value', () => {
    const { result } = renderHook(() => useCounter(10));
    
    act(() => {
      result.current.increment();
      result.current.increment();
    });
    
    expect(result.current.count).toBe(12);
    
    act(() => {
      result.current.reset();
    });
    
    expect(result.current.count).toBe(10);
  });
});
```

### Testing Hooks with Effects
```javascript
import { renderHook, waitFor } from '@testing-library/react';

// useFetch.js
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!url) return;

    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(url);
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
    };

    fetchData();
  }, [url]);

  return { data, loading, error };
}

// useFetch.test.js
// Mock fetch globally
global.fetch = jest.fn();

describe('useFetch', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  test('initial state', () => {
    const { result } = renderHook(() => useFetch(''));
    
    expect(result.current.data).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  test('successful fetch', async () => {
    const mockData = { id: 1, name: 'Test' };
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockData,
    });

    const { result } = renderHook(() => useFetch('/api/test'));

    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toEqual(mockData);
    expect(result.current.error).toBeNull();
    expect(fetch).toHaveBeenCalledWith('/api/test');
  });

  test('fetch error', async () => {
    fetch.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useFetch('/api/test'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe('Network error');
  });

  test('HTTP error', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
    });

    const { result } = renderHook(() => useFetch('/api/test'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('HTTP error! status: 404');
  });
});
```

## Integration Testing

### Testing Component Integration
```javascript
import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';

// TodoApp.js - Integration of multiple components
function TodoApp() {
  const [todos, setTodos] = useState([]);

  const addTodo = (text) => {
    setTodos([...todos, { id: Date.now(), text, completed: false }]);
  };

  const toggleTodo = (id) => {
    setTodos(todos.map(todo => 
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };

  const deleteTodo = (id) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  return (
    <div>
      <h1>Todo App</h1>
      <TodoForm onSubmit={addTodo} />
      <TodoList 
        todos={todos} 
        onToggle={toggleTodo} 
        onDelete={deleteTodo} 
      />
      <TodoStats todos={todos} />
    </div>
  );
}

// TodoApp.test.js
describe('TodoApp Integration', () => {
  test('complete todo workflow', async () => {
    const user = userEvent.setup();
    
    render(<TodoApp />);

    // Add first todo
    const input = screen.getByPlaceholderText(/add a todo/i);
    await user.type(input, 'Learn React Testing');
    await user.click(screen.getByText(/add todo/i));

    expect(screen.getByText('Learn React Testing')).toBeInTheDocument();
    expect(screen.getByText(/total: 1/i)).toBeInTheDocument();
    expect(screen.getByText(/completed: 0/i)).toBeInTheDocument();

    // Add second todo
    await user.clear(input);
    await user.type(input, 'Write tests');
    await user.click(screen.getByText(/add todo/i));

    expect(screen.getByText('Write tests')).toBeInTheDocument();
    expect(screen.getByText(/total: 2/i)).toBeInTheDocument();

    // Complete first todo
    const firstTodo = screen.getByLabelText('Learn React Testing');
    await user.click(firstTodo);

    expect(firstTodo).toBeChecked();
    expect(screen.getByText(/completed: 1/i)).toBeInTheDocument();

    // Delete second todo
    const deleteButton = screen.getAllByText(/delete/i)[1];
    await user.click(deleteButton);

    expect(screen.queryByText('Write tests')).not.toBeInTheDocument();
    expect(screen.getByText(/total: 1/i)).toBeInTheDocument();
  });

  test('handles empty state', () => {
    render(<TodoApp />);
    
    expect(screen.getByText(/no todos yet/i)).toBeInTheDocument();
    expect(screen.getByText(/total: 0/i)).toBeInTheDocument();
  });
});
```

### Testing with Router
```javascript
import React from 'react';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import App from './App';

function renderWithRouter(ui, { initialEntries = ['/'] } = {}) {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      {ui}
    </MemoryRouter>
  );
}

describe('App Routing', () => {
  test('renders home page by default', () => {
    renderWithRouter(<App />);
    
    expect(screen.getByText(/welcome to home/i)).toBeInTheDocument();
  });

  test('navigates to about page', async () => {
    const user = userEvent.setup();
    
    renderWithRouter(<App />);
    
    await user.click(screen.getByText(/about/i));
    
    expect(screen.getByText(/about us/i)).toBeInTheDocument();
  });

  test('renders 404 for unknown route', () => {
    renderWithRouter(<App />, { initialEntries: ['/unknown-route'] });
    
    expect(screen.getByText(/page not found/i)).toBeInTheDocument();
  });

  test('protected route redirects when not authenticated', () => {
    renderWithRouter(<App />, { initialEntries: ['/dashboard'] });
    
    expect(screen.getByText(/please log in/i)).toBeInTheDocument();
  });
});
```

## Mocking

### Mocking API Calls
```javascript
import axios from 'axios';
import { render, screen, waitFor } from '@testing-library/react';
import UserList from './UserList';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('UserList', () => {
  beforeEach(() => {
    mockedAxios.get.mockClear();
  });

  test('displays users after successful fetch', async () => {
    const users = [
      { id: 1, name: 'John Doe', email: 'john@example.com' },
      { id: 2, name: 'Jane Smith', email: 'jane@example.com' },
    ];

    mockedAxios.get.mockResolvedValueOnce({ data: users });

    render(<UserList />);

    expect(screen.getByText(/loading/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });

    expect(screen.getByText('Jane Smith')).toBeInTheDocument();
    expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
  });

  test('displays error message on fetch failure', async () => {
    mockedAxios.get.mockRejectedValueOnce(new Error('Network Error'));

    render(<UserList />);

    await waitFor(() => {
      expect(screen.getByText(/error loading users/i)).toBeInTheDocument();
    });
  });
});
```

### Mocking Modules and Dependencies
```javascript
// Mock external dependencies
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => jest.fn(),
  useParams: () => ({ id: '123' }),
}));

// Mock local modules
jest.mock('./utils/api', () => ({
  fetchUser: jest.fn(),
  updateUser: jest.fn(),
}));

// Mock hooks
jest.mock('./hooks/useAuth', () => ({
  useAuth: () => ({
    user: { id: 1, name: 'Test User' },
    isAuthenticated: true,
    login: jest.fn(),
    logout: jest.fn(),
  }),
}));

// Partial mocking
jest.mock('./utils/helpers', () => ({
  ...jest.requireActual('./utils/helpers'),
  formatDate: jest.fn(() => '2023-01-01'),
}));
```

### Mocking Browser APIs
```javascript
// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// Mock window.location
delete window.location;
window.location = { href: 'http://localhost:3000' };

// Mock fetch
global.fetch = jest.fn();

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn(() => ({
  observe: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});
```

## E2E Testing

### Cypress Setup
```javascript
// cypress/support/commands.js
Cypress.Commands.add('login', (email, password) => {
  cy.visit('/login');
  cy.get('[data-testid=email-input]').type(email);
  cy.get('[data-testid=password-input]').type(password);
  cy.get('[data-testid=submit-button]').click();
});

Cypress.Commands.add('createTodo', (text) => {
  cy.get('[data-testid=todo-input]').type(text);
  cy.get('[data-testid=add-button]').click();
});

// cypress/integration/todo-app.spec.js
describe('Todo App E2E', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  it('should add and complete a todo', () => {
    cy.createTodo('Learn Cypress');
    
    cy.contains('Learn Cypress').should('be.visible');
    cy.get('[data-testid=todo-count]').should('contain', '1');
    
    cy.get('[data-testid=todo-checkbox]').first().click();
    cy.get('[data-testid=completed-count]').should('contain', '1');
  });

  it('should filter todos', () => {
    cy.createTodo('Active todo');
    cy.createTodo('Completed todo');
    
    cy.get('[data-testid=todo-checkbox]').last().click();
    
    cy.get('[data-testid=filter-active]').click();
    cy.contains('Completed todo').should('not.exist');
    cy.contains('Active todo').should('be.visible');
    
    cy.get('[data-testid=filter-completed]').click();
    cy.contains('Active todo').should('not.exist');
    cy.contains('Completed todo').should('be.visible');
  });

  it('should persist todos after page refresh', () => {
    cy.createTodo('Persistent todo');
    cy.reload();
    cy.contains('Persistent todo').should('be.visible');
  });
});
```

### Playwright Testing
```javascript
// tests/todo-app.spec.js
import { test, expect } from '@playwright/test';

test.describe('Todo App', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should add a new todo', async ({ page }) => {
    await page.fill('[data-testid=todo-input]', 'Learn Playwright');
    await page.click('[data-testid=add-button]');
    
    await expect(page.locator('text=Learn Playwright')).toBeVisible();
    await expect(page.locator('[data-testid=todo-count]')).toHaveText('1');
  });

  test('should complete a todo', async ({ page }) => {
    await page.fill('[data-testid=todo-input]', 'Complete me');
    await page.click('[data-testid=add-button]');
    
    await page.click('[data-testid=todo-checkbox]');
    
    await expect(page.locator('[data-testid=completed-count]')).toHaveText('1');
    await expect(page.locator('.todo-item')).toHaveClass(/completed/);
  });

  test('should work on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    
    await page.fill('[data-testid=todo-input]', 'Mobile todo');
    await page.click('[data-testid=add-button]');
    
    await expect(page.locator('text=Mobile todo')).toBeVisible();
  });
});
```

## Testing Best Practices

### Writing Maintainable Tests
```javascript
// ❌ Bad: Testing implementation details
test('bad test', () => {
  const wrapper = shallow(<Counter />);
  expect(wrapper.state('count')).toBe(0);
  wrapper.instance().increment();
  expect(wrapper.state('count')).toBe(1);
});

// ✅ Good: Testing behavior
test('good test', async () => {
  const user = userEvent.setup();
  render(<Counter />);
  
  expect(screen.getByText('Count: 0')).toBeInTheDocument();
  
  await user.click(screen.getByText('Increment'));
  
  expect(screen.getByText('Count: 1')).toBeInTheDocument();
});

// ❌ Bad: Too many implementation details
test('bad form test', async () => {
  const user = userEvent.setup();
  const { container } = render(<LoginForm />);
  
  const emailInput = container.querySelector('input[type="email"]');
  const passwordInput = container.querySelector('input[type="password"]');
  
  await user.type(emailInput, 'test@example.com');
  await user.type(passwordInput, 'password');
});

// ✅ Good: Focus on user interaction
test('good form test', async () => {
  const user = userEvent.setup();
  render(<LoginForm />);
  
  await user.type(screen.getByLabelText(/email/i), 'test@example.com');
  await user.type(screen.getByLabelText(/password/i), 'password');
});
```

### Test Organization
```javascript
// Use descriptive test names
describe('UserProfile Component', () => {
  describe('when user is not logged in', () => {
    test('displays login prompt', () => {
      // test implementation
    });

    test('redirects to login page when profile button clicked', () => {
      // test implementation
    });
  });

  describe('when user is logged in', () => {
    test('displays user name and email', () => {
      // test implementation
    });

    test('allows editing profile information', () => {
      // test implementation
    });
  });
});

// Create test utilities
// test-utils.js
import React from 'react';
import { render } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from './ThemeContext';
import { AuthProvider } from './AuthContext';

function AllTheProviders({ children }) {
  return (
    <BrowserRouter>
      <AuthProvider>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}

const customRender = (ui, options) =>
  render(ui, { wrapper: AllTheProviders, ...options });

export * from '@testing-library/react';
export { customRender as render };
```

### Performance Testing
```javascript
// Performance testing with React Testing Library
import { render, screen } from '@testing-library/react';
import { act } from 'react-dom/test-utils';

test('renders large list efficiently', () => {
  const items = Array.from({ length: 10000 }, (_, i) => ({ id: i, name: `Item ${i}` }));
  
  const startTime = performance.now();
  
  act(() => {
    render(<VirtualList items={items} />);
  });
  
  const endTime = performance.now();
  const renderTime = endTime - startTime;
  
  expect(renderTime).toBeLessThan(100); // Should render in less than 100ms
  expect(screen.getByText('Item 0')).toBeInTheDocument();
});

// Memory leak testing
test('cleans up properly on unmount', () => {
  const { unmount } = render(<ComponentWithSubscriptions />);
  
  // Mock memory monitoring
  const initialMemory = performance.memory?.usedJSHeapSize;
  
  unmount();
  
  // Force garbage collection if available
  if (global.gc) {
    global.gc();
  }
  
  const finalMemory = performance.memory?.usedJSHeapSize;
  
  // Memory should not increase significantly
  expect(finalMemory - initialMemory).toBeLessThan(1000000); // 1MB threshold
});
```

---

*Continue to: [08-react-advanced.md](./08-react-advanced.md)*
