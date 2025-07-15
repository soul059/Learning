# React Components

## Table of Contents
- [Component Types](#component-types)
- [Props Deep Dive](#props-deep-dive)
- [State Management](#state-management)
- [Event Handling](#event-handling)
- [Component Lifecycle](#component-lifecycle)
- [Component Composition](#component-composition)
- [Higher-Order Components (HOCs)](#higher-order-components-hocs)
- [Render Props Pattern](#render-props-pattern)
- [Error Boundaries](#error-boundaries)

## Component Types

### Function Components (Modern Approach)
```javascript
// Basic function component
function Welcome(props) {
  return <h1>Hello, {props.name}!</h1>;
}

// Arrow function component
const Welcome = (props) => {
  return <h1>Hello, {props.name}!</h1>;
};

// Destructured props
const Welcome = ({ name, age, isVip }) => {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      {age && <p>Age: {age}</p>}
      {isVip && <span className="vip-badge">VIP</span>}
    </div>
  );
};
```

### Class Components (Legacy but still important)
```javascript
import React, { Component } from 'react';

class Welcome extends Component {
  constructor(props) {
    super(props);
    this.state = {
      count: 0,
      isVisible: true
    };
  }

  handleClick = () => {
    this.setState({ count: this.state.count + 1 });
  };

  render() {
    const { name } = this.props;
    const { count, isVisible } = this.state;

    return (
      <div>
        <h1>Hello, {name}!</h1>
        <p>Count: {count}</p>
        <button onClick={this.handleClick}>Increment</button>
      </div>
    );
  }
}
```

### Component Conversion (Class to Function)
```javascript
// Class component
class Counter extends Component {
  constructor(props) {
    super(props);
    this.state = { count: 0 };
  }

  componentDidMount() {
    document.title = `Count: ${this.state.count}`;
  }

  componentDidUpdate() {
    document.title = `Count: ${this.state.count}`;
  }

  render() {
    return (
      <div>
        <p>Count: {this.state.count}</p>
        <button onClick={() => this.setState({ count: this.state.count + 1 })}>
          +
        </button>
      </div>
    );
  }
}

// Equivalent function component
import { useState, useEffect } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    document.title = `Count: ${count}`;
  }, [count]);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>+</button>
    </div>
  );
}
```

## Props Deep Dive

### Basic Props Usage
```javascript
function UserCard({ user, showEmail = false, onEdit }) {
  return (
    <div className="user-card">
      <img src={user.avatar} alt={`${user.name}'s avatar`} />
      <h3>{user.name}</h3>
      {showEmail && <p>{user.email}</p>}
      <button onClick={() => onEdit(user.id)}>Edit</button>
    </div>
  );
}

// Usage
function App() {
  const handleEdit = (userId) => {
    console.log('Editing user:', userId);
  };

  return (
    <UserCard 
      user={{ id: 1, name: 'John', email: 'john@example.com', avatar: 'avatar.jpg' }}
      showEmail={true}
      onEdit={handleEdit}
    />
  );
}
```

### Props Validation with PropTypes
```javascript
import PropTypes from 'prop-types';

function UserCard({ user, showEmail, onEdit }) {
  // Component implementation
}

UserCard.propTypes = {
  user: PropTypes.shape({
    id: PropTypes.number.isRequired,
    name: PropTypes.string.isRequired,
    email: PropTypes.string.isRequired,
    avatar: PropTypes.string
  }).isRequired,
  showEmail: PropTypes.bool,
  onEdit: PropTypes.func.isRequired
};

UserCard.defaultProps = {
  showEmail: false
};
```

### TypeScript Props (Recommended)
```typescript
interface User {
  id: number;
  name: string;
  email: string;
  avatar?: string;
}

interface UserCardProps {
  user: User;
  showEmail?: boolean;
  onEdit: (userId: number) => void;
}

const UserCard: React.FC<UserCardProps> = ({ 
  user, 
  showEmail = false, 
  onEdit 
}) => {
  return (
    <div className="user-card">
      <img src={user.avatar || '/default-avatar.png'} alt={`${user.name}'s avatar`} />
      <h3>{user.name}</h3>
      {showEmail && <p>{user.email}</p>}
      <button onClick={() => onEdit(user.id)}>Edit</button>
    </div>
  );
};
```

### Props Spreading and Rest
```javascript
function Button({ children, className, ...restProps }) {
  return (
    <button 
      className={`btn ${className || ''}`}
      {...restProps}
    >
      {children}
    </button>
  );
}

// Usage
<Button 
  className="primary" 
  onClick={handleClick}
  disabled={isLoading}
  type="submit"
>
  Submit
</Button>
```

### Children Prop Patterns
```javascript
// Basic children
function Card({ children, title }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <div className="card-content">
        {children}
      </div>
    </div>
  );
}

// Function as children (render prop)
function DataFetcher({ children, url }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      });
  }, [url]);

  return children({ data, loading });
}

// Usage
<DataFetcher url="/api/users">
  {({ data, loading }) => (
    loading ? <div>Loading...</div> : <UserList users={data} />
  )}
</DataFetcher>

// Named children slots
function Layout({ header, sidebar, children }) {
  return (
    <div className="layout">
      <header>{header}</header>
      <div className="layout-body">
        <aside>{sidebar}</aside>
        <main>{children}</main>
      </div>
    </div>
  );
}

// Usage
<Layout
  header={<Header />}
  sidebar={<Sidebar />}
>
  <MainContent />
</Layout>
```

## State Management

### useState Hook
```javascript
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  // Multiple state variables
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Object state
  const [user, setUser] = useState({
    name: '',
    email: '',
    age: 0
  });

  // Array state
  const [todos, setTodos] = useState([]);

  // Function to update object state
  const updateUser = (field, value) => {
    setUser(prevUser => ({
      ...prevUser,
      [field]: value
    }));
  };

  // Function to update array state
  const addTodo = (text) => {
    setTodos(prevTodos => [...prevTodos, {
      id: Date.now(),
      text,
      completed: false
    }]);
  };

  const toggleTodo = (id) => {
    setTodos(prevTodos =>
      prevTodos.map(todo =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    );
  };

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>+</button>
      
      <input 
        value={user.name}
        onChange={(e) => updateUser('name', e.target.value)}
        placeholder="Name"
      />
    </div>
  );
}
```

### State Patterns
```javascript
// Lazy initial state (expensive computation)
const [data, setData] = useState(() => {
  const savedData = localStorage.getItem('data');
  return savedData ? JSON.parse(savedData) : [];
});

// Previous state pattern
const [count, setCount] = useState(0);

const increment = () => {
  setCount(prevCount => prevCount + 1); // Safer for async updates
};

// Multiple related states with useReducer
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
    default:
      return state;
  }
}

function TodoApp() {
  const [todos, dispatch] = useReducer(todoReducer, []);

  return (
    <div>
      <button onClick={() => dispatch({ type: 'ADD_TODO', text: 'New todo' })}>
        Add Todo
      </button>
      {todos.map(todo => (
        <div key={todo.id}>
          <span 
            onClick={() => dispatch({ type: 'TOGGLE_TODO', id: todo.id })}
          >
            {todo.text}
          </span>
          <button onClick={() => dispatch({ type: 'DELETE_TODO', id: todo.id })}>
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}
```

## Event Handling

### Basic Event Handling
```javascript
function Form() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });

  // Input change handler
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Form submitted:', formData);
  };

  // Button click with parameters
  const handleButtonClick = (action, id) => {
    console.log(`Action: ${action}, ID: ${id}`);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input 
        type="text"
        name="name"
        value={formData.name}
        onChange={handleInputChange}
        placeholder="Name"
      />
      
      <input 
        type="email"
        name="email"
        value={formData.email}
        onChange={handleInputChange}
        placeholder="Email"
      />
      
      <textarea 
        name="message"
        value={formData.message}
        onChange={handleInputChange}
        placeholder="Message"
      />
      
      <button type="submit">Submit</button>
      
      <button 
        type="button"
        onClick={() => handleButtonClick('edit', 123)}
      >
        Edit Item
      </button>
    </form>
  );
}
```

### Advanced Event Patterns
```javascript
function AdvancedEvents() {
  const [items, setItems] = useState([]);

  // Event delegation pattern
  const handleListClick = (e) => {
    if (e.target.matches('.delete-btn')) {
      const id = e.target.dataset.id;
      setItems(prev => prev.filter(item => item.id !== id));
    } else if (e.target.matches('.edit-btn')) {
      const id = e.target.dataset.id;
      // Edit logic
    }
  };

  // Debounced input
  const [searchTerm, setSearchTerm] = useState('');
  const debouncedSearch = useCallback(
    debounce((term) => {
      // Perform search
      console.log('Searching for:', term);
    }, 300),
    []
  );

  useEffect(() => {
    debouncedSearch(searchTerm);
  }, [searchTerm, debouncedSearch]);

  // Keyboard events
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleSubmit();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  return (
    <div>
      <input 
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Search..."
      />
      
      <ul onClick={handleListClick}>
        {items.map(item => (
          <li key={item.id}>
            {item.name}
            <button className="edit-btn" data-id={item.id}>Edit</button>
            <button className="delete-btn" data-id={item.id}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Debounce utility
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
```

## Component Lifecycle

### Function Component Lifecycle (with useEffect)
```javascript
import { useState, useEffect } from 'react';

function ComponentLifecycle({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // ComponentDidMount equivalent
  useEffect(() => {
    console.log('Component mounted');
    
    // Cleanup function (ComponentWillUnmount equivalent)
    return () => {
      console.log('Component will unmount');
    };
  }, []); // Empty dependency array = run once on mount

  // ComponentDidUpdate equivalent
  useEffect(() => {
    console.log('Component updated');
  }); // No dependency array = run on every render

  // Watch specific values
  useEffect(() => {
    if (userId) {
      setLoading(true);
      fetchUser(userId)
        .then(userData => {
          setUser(userData);
          setLoading(false);
        })
        .catch(error => {
          console.error('Failed to fetch user:', error);
          setLoading(false);
        });
    }
  }, [userId]); // Run when userId changes

  // Multiple effects for separation of concerns
  useEffect(() => {
    document.title = user ? `User: ${user.name}` : 'Loading...';
  }, [user]);

  useEffect(() => {
    const handleResize = () => {
      console.log('Window resized');
    };

    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
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

### Class Component Lifecycle (Reference)
```javascript
class ComponentLifecycle extends Component {
  constructor(props) {
    super(props);
    this.state = { data: null, loading: true };
    console.log('Constructor');
  }

  static getDerivedStateFromProps(props, state) {
    console.log('getDerivedStateFromProps');
    return null; // No state update
  }

  componentDidMount() {
    console.log('componentDidMount');
    this.fetchData();
  }

  componentDidUpdate(prevProps, prevState) {
    console.log('componentDidUpdate');
    if (prevProps.userId !== this.props.userId) {
      this.fetchData();
    }
  }

  componentWillUnmount() {
    console.log('componentWillUnmount');
    // Cleanup subscriptions, timers, etc.
  }

  shouldComponentUpdate(nextProps, nextState) {
    console.log('shouldComponentUpdate');
    return true; // Return false to prevent re-render
  }

  getSnapshotBeforeUpdate(prevProps, prevState) {
    console.log('getSnapshotBeforeUpdate');
    return null;
  }

  componentDidCatch(error, errorInfo) {
    console.log('componentDidCatch', error, errorInfo);
  }

  render() {
    console.log('render');
    return <div>{this.state.data}</div>;
  }
}
```

## Component Composition

### Basic Composition
```javascript
function Button({ variant = 'primary', size = 'medium', children, ...props }) {
  const className = `btn btn-${variant} btn-${size}`;
  
  return (
    <button className={className} {...props}>
      {children}
    </button>
  );
}

function IconButton({ icon, children, ...props }) {
  return (
    <Button {...props}>
      <span className="icon">{icon}</span>
      {children}
    </Button>
  );
}

function LoadingButton({ loading, children, ...props }) {
  return (
    <Button disabled={loading} {...props}>
      {loading ? <Spinner /> : children}
    </Button>
  );
}
```

### Compound Components
```javascript
function Tabs({ children, defaultActiveTab = 0 }) {
  const [activeTab, setActiveTab] = useState(defaultActiveTab);

  return (
    <div className="tabs">
      {React.Children.map(children, (child, index) =>
        React.cloneElement(child, {
          isActive: index === activeTab,
          onActivate: () => setActiveTab(index),
          index
        })
      )}
    </div>
  );
}

function Tab({ children, label, isActive, onActivate }) {
  return (
    <div className="tab">
      <button 
        className={`tab-button ${isActive ? 'active' : ''}`}
        onClick={onActivate}
      >
        {label}
      </button>
      {isActive && (
        <div className="tab-content">
          {children}
        </div>
      )}
    </div>
  );
}

// Usage
<Tabs defaultActiveTab={0}>
  <Tab label="Tab 1">Content for tab 1</Tab>
  <Tab label="Tab 2">Content for tab 2</Tab>
  <Tab label="Tab 3">Content for tab 3</Tab>
</Tabs>
```

### Render Props and Component Injection
```javascript
function Modal({ children, renderTrigger, isOpen: controlledOpen, onClose }) {
  const [internalOpen, setInternalOpen] = useState(false);
  const isOpen = controlledOpen !== undefined ? controlledOpen : internalOpen;
  const setIsOpen = controlledOpen !== undefined ? onClose : setInternalOpen;

  return (
    <>
      {renderTrigger({ openModal: () => setIsOpen(true) })}
      {isOpen && (
        <div className="modal-overlay" onClick={() => setIsOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setIsOpen(false)}>
              ×
            </button>
            {children}
          </div>
        </div>
      )}
    </>
  );
}

// Usage
<Modal renderTrigger={({ openModal }) => (
  <button onClick={openModal}>Open Modal</button>
)}>
  <h2>Modal Content</h2>
  <p>This is inside the modal!</p>
</Modal>
```

## Higher-Order Components (HOCs)

```javascript
// Basic HOC
function withLoading(WrappedComponent) {
  return function WithLoadingComponent(props) {
    if (props.loading) {
      return <div>Loading...</div>;
    }
    return <WrappedComponent {...props} />;
  };
}

// Enhanced HOC with additional props
function withAuth(WrappedComponent) {
  return function WithAuthComponent(props) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
      checkAuthStatus()
        .then(userData => {
          setUser(userData);
          setLoading(false);
        })
        .catch(() => {
          setUser(null);
          setLoading(false);
        });
    }, []);

    if (loading) return <div>Checking authentication...</div>;
    if (!user) return <div>Please log in</div>;

    return <WrappedComponent {...props} user={user} />;
  };
}

// Usage
const ProtectedProfile = withAuth(withLoading(UserProfile));

// Modern alternative: Custom Hook
function useAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus()
      .then(userData => {
        setUser(userData);
        setLoading(false);
      })
      .catch(() => {
        setUser(null);
        setLoading(false);
      });
  }, []);

  return { user, loading };
}

function UserProfile() {
  const { user, loading } = useAuth();

  if (loading) return <div>Loading...</div>;
  if (!user) return <div>Please log in</div>;

  return <div>Welcome, {user.name}!</div>;
}
```

## Render Props Pattern

```javascript
function MouseTracker({ children }) {
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    document.addEventListener('mousemove', handleMouseMove);
    return () => document.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return children(position);
}

// Usage
<MouseTracker>
  {({ x, y }) => (
    <div>
      Mouse position: {x}, {y}
    </div>
  )}
</MouseTracker>

// Generic data fetcher with render props
function DataFetcher({ url, children }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(url)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err);
        setLoading(false);
      });
  }, [url]);

  return children({ data, loading, error });
}

// Usage
<DataFetcher url="/api/users">
  {({ data, loading, error }) => {
    if (loading) return <Spinner />;
    if (error) return <ErrorMessage error={error} />;
    return <UserList users={data} />;
  }}
</DataFetcher>
```

## Error Boundaries

```javascript
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error,
      errorInfo
    });
    
    // Log error to reporting service
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          {this.props.fallback ? (
            this.props.fallback(this.state.error, this.state.errorInfo)
          ) : (
            <details>
              <summary>Error Details</summary>
              <pre>{this.state.error && this.state.error.toString()}</pre>
              <pre>{this.state.errorInfo.componentStack}</pre>
            </details>
          )}
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            Try Again
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
    <ErrorBoundary fallback={(error, errorInfo) => (
      <div>
        <h2>Oops! Something went wrong</h2>
        <p>We're sorry for the inconvenience.</p>
      </div>
    )}>
      <Header />
      <MainContent />
      <Footer />
    </ErrorBoundary>
  );
}

// Function component alternative (React 18+)
function ErrorBoundaryHook({ children, fallback }) {
  return (
    <ErrorBoundary fallback={fallback}>
      {children}
    </ErrorBoundary>
  );
}
```

---

*Continue to: [03-react-hooks.md](./03-react-hooks.md)*
