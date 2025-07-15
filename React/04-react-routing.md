# React Router

## Table of Contents
- [Introduction to React Router](#introduction-to-react-router)
- [Setup and Installation](#setup-and-installation)
- [Basic Routing](#basic-routing)
- [Route Parameters](#route-parameters)
- [Nested Routes](#nested-routes)
- [Navigation](#navigation)
- [Route Guards and Protection](#route-guards-and-protection)
- [Advanced Routing Patterns](#advanced-routing-patterns)
- [Data Loading](#data-loading)
- [Error Handling](#error-handling)

## Introduction to React Router

React Router is the standard routing library for React applications. It enables navigation between different components/pages without full page reloads, creating a Single Page Application (SPA) experience.

### Key Concepts:
- **Router**: Provides routing context to the app
- **Routes**: Define which component to render for each URL
- **Route**: Individual route definition
- **Link**: Navigation component for internal links
- **useNavigate**: Hook for programmatic navigation
- **useParams**: Hook to access URL parameters
- **useLocation**: Hook to access current location

## Setup and Installation

### Installation
```bash
# Install React Router
npm install react-router-dom

# For TypeScript
npm install @types/react-router-dom
```

### Basic Setup
```javascript
// main.jsx (or index.js)
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
```

### Router Types
```javascript
import {
  BrowserRouter,    // Uses HTML5 history API (recommended)
  HashRouter,       // Uses hash portion of URL
  MemoryRouter,     // For testing or React Native
  StaticRouter      // For server-side rendering
} from 'react-router-dom';

// BrowserRouter (most common)
function App() {
  return (
    <BrowserRouter>
      {/* Your app routes */}
    </BrowserRouter>
  );
}

// HashRouter (for static hosting)
function App() {
  return (
    <HashRouter>
      {/* URLs will be like: example.com/#/about */}
    </HashRouter>
  );
}
```

## Basic Routing

### Simple Route Setup
```javascript
import { Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import About from './components/About';
import Contact from './components/Contact';
import NotFound from './components/NotFound';

function App() {
  return (
    <div className="App">
      <nav>
        <Link to="/">Home</Link>
        <Link to="/about">About</Link>
        <Link to="/contact">Contact</Link>
      </nav>

      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}
```

### Route Components
```javascript
// Home.jsx
function Home() {
  return (
    <div>
      <h1>Welcome to Home Page</h1>
      <p>This is the homepage content.</p>
    </div>
  );
}

// About.jsx
function About() {
  return (
    <div>
      <h1>About Us</h1>
      <p>Learn more about our company.</p>
    </div>
  );
}

// NotFound.jsx
function NotFound() {
  return (
    <div>
      <h1>404 - Page Not Found</h1>
      <p>The page you're looking for doesn't exist.</p>
      <Link to="/">Go back to Home</Link>
    </div>
  );
}
```

### Index Routes
```javascript
import { Routes, Route } from 'react-router-dom';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        {/* Index route - renders when parent path matches exactly */}
        <Route index element={<Home />} />
        <Route path="about" element={<About />} />
        <Route path="contact" element={<Contact />} />
      </Route>
    </Routes>
  );
}

function Layout() {
  return (
    <div>
      <nav>
        <Link to="/">Home</Link>
        <Link to="/about">About</Link>
        <Link to="/contact">Contact</Link>
      </nav>
      <main>
        <Outlet /> {/* Renders child routes */}
      </main>
    </div>
  );
}
```

## Route Parameters

### URL Parameters
```javascript
import { Routes, Route } from 'react-router-dom';
import { useParams } from 'react-router-dom';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/users/:userId" element={<UserProfile />} />
      <Route path="/users/:userId/posts/:postId" element={<UserPost />} />
      <Route path="/products/:category/:productId" element={<Product />} />
    </Routes>
  );
}

// UserProfile component
function UserProfile() {
  const { userId } = useParams();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUser(userId)
      .then(userData => {
        setUser(userData);
        setLoading(false);
      })
      .catch(error => {
        console.error('Failed to fetch user:', error);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <div>Loading user...</div>;
  if (!user) return <div>User not found</div>;

  return (
    <div>
      <h1>{user.name}</h1>
      <p>Email: {user.email}</p>
      <p>User ID: {userId}</p>
    </div>
  );
}

// UserPost component
function UserPost() {
  const { userId, postId } = useParams();
  
  return (
    <div>
      <h1>Post {postId} by User {userId}</h1>
      {/* Post content */}
    </div>
  );
}

// Product component with multiple parameters
function Product() {
  const { category, productId } = useParams();
  
  return (
    <div>
      <h1>Product {productId}</h1>
      <p>Category: {category}</p>
    </div>
  );
}
```

### Optional Parameters
```javascript
function App() {
  return (
    <Routes>
      {/* Optional parameter with ? */}
      <Route path="/users/:userId/:tab?" element={<UserProfile />} />
      {/* Multiple optional parameters */}
      <Route path="/search/:query?/:filter?/:sort?" element={<SearchResults />} />
    </Routes>
  );
}

function UserProfile() {
  const { userId, tab = 'profile' } = useParams();
  
  return (
    <div>
      <h1>User {userId}</h1>
      <nav>
        <Link to={`/users/${userId}`}>Profile</Link>
        <Link to={`/users/${userId}/posts`}>Posts</Link>
        <Link to={`/users/${userId}/settings`}>Settings</Link>
      </nav>
      
      {tab === 'profile' && <ProfileTab />}
      {tab === 'posts' && <PostsTab />}
      {tab === 'settings' && <SettingsTab />}
    </div>
  );
}
```

### Query Parameters
```javascript
import { useSearchParams } from 'react-router-dom';

function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  
  // Get query parameters
  const query = searchParams.get('q') || '';
  const category = searchParams.get('category') || 'all';
  const page = parseInt(searchParams.get('page')) || 1;
  
  const handleSearch = (newQuery) => {
    setSearchParams({
      q: newQuery,
      category,
      page: 1 // Reset to first page
    });
  };
  
  const handleCategoryChange = (newCategory) => {
    setSearchParams({
      q: query,
      category: newCategory,
      page: 1
    });
  };
  
  const handlePageChange = (newPage) => {
    setSearchParams({
      q: query,
      category,
      page: newPage
    });
  };

  return (
    <div>
      <h1>Search Results</h1>
      
      <input 
        value={query}
        onChange={(e) => handleSearch(e.target.value)}
        placeholder="Search..."
      />
      
      <select 
        value={category} 
        onChange={(e) => handleCategoryChange(e.target.value)}
      >
        <option value="all">All Categories</option>
        <option value="electronics">Electronics</option>
        <option value="clothing">Clothing</option>
      </select>
      
      <div>
        <p>Query: {query}</p>
        <p>Category: {category}</p>
        <p>Page: {page}</p>
      </div>
      
      {/* Pagination */}
      <div>
        <button 
          onClick={() => handlePageChange(page - 1)}
          disabled={page <= 1}
        >
          Previous
        </button>
        <span>Page {page}</span>
        <button onClick={() => handlePageChange(page + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}
```

## Nested Routes

### Basic Nested Routes
```javascript
import { Routes, Route, Outlet } from 'react-router-dom';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="about" element={<About />} />
        
        {/* Nested routes for dashboard */}
        <Route path="dashboard" element={<DashboardLayout />}>
          <Route index element={<DashboardHome />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="users" element={<Users />} />
          <Route path="settings" element={<Settings />} />
        </Route>
        
        {/* Nested routes for shop */}
        <Route path="shop" element={<ShopLayout />}>
          <Route index element={<ProductList />} />
          <Route path="categories" element={<Categories />} />
          <Route path="products/:productId" element={<ProductDetail />} />
          <Route path="cart" element={<Cart />} />
        </Route>
      </Route>
    </Routes>
  );
}

// Layout component
function Layout() {
  return (
    <div>
      <header>
        <nav>
          <Link to="/">Home</Link>
          <Link to="/about">About</Link>
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/shop">Shop</Link>
        </nav>
      </header>
      <main>
        <Outlet /> {/* Renders child routes */}
      </main>
    </div>
  );
}

// Dashboard layout with its own navigation
function DashboardLayout() {
  return (
    <div className="dashboard">
      <aside>
        <nav>
          <Link to="/dashboard">Overview</Link>
          <Link to="/dashboard/analytics">Analytics</Link>
          <Link to="/dashboard/users">Users</Link>
          <Link to="/dashboard/settings">Settings</Link>
        </nav>
      </aside>
      <div className="dashboard-content">
        <Outlet />
      </div>
    </div>
  );
}
```

### Deep Nesting Example
```javascript
function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route path="admin" element={<AdminLayout />}>
          <Route path="users" element={<UserManagement />}>
            <Route index element={<UserList />} />
            <Route path=":userId" element={<UserDetail />}>
              <Route index element={<UserProfile />} />
              <Route path="edit" element={<UserEdit />} />
              <Route path="permissions" element={<UserPermissions />} />
            </Route>
            <Route path="new" element={<CreateUser />} />
          </Route>
        </Route>
      </Route>
    </Routes>
  );
}

function UserManagement() {
  return (
    <div>
      <h1>User Management</h1>
      <nav>
        <Link to="/admin/users">All Users</Link>
        <Link to="/admin/users/new">Create User</Link>
      </nav>
      <Outlet />
    </div>
  );
}

function UserDetail() {
  const { userId } = useParams();
  
  return (
    <div>
      <h2>User {userId}</h2>
      <nav>
        <Link to={`/admin/users/${userId}`}>Profile</Link>
        <Link to={`/admin/users/${userId}/edit`}>Edit</Link>
        <Link to={`/admin/users/${userId}/permissions`}>Permissions</Link>
      </nav>
      <Outlet />
    </div>
  );
}
```

## Navigation

### Link Component
```javascript
import { Link } from 'react-router-dom';

function Navigation() {
  return (
    <nav>
      {/* Basic links */}
      <Link to="/">Home</Link>
      <Link to="/about">About</Link>
      
      {/* Link with state */}
      <Link 
        to="/profile" 
        state={{ from: 'navigation' }}
      >
        Profile
      </Link>
      
      {/* Link with custom styling */}
      <Link 
        to="/products" 
        className="nav-link"
        style={{ color: 'blue' }}
      >
        Products
      </Link>
      
      {/* Conditional link */}
      {user ? (
        <Link to="/dashboard">Dashboard</Link>
      ) : (
        <Link to="/login">Login</Link>
      )}
    </nav>
  );
}
```

### NavLink Component (Active Styling)
```javascript
import { NavLink } from 'react-router-dom';

function Navigation() {
  return (
    <nav>
      <NavLink 
        to="/"
        className={({ isActive }) => 
          isActive ? 'nav-link active' : 'nav-link'
        }
      >
        Home
      </NavLink>
      
      <NavLink 
        to="/about"
        style={({ isActive }) => ({
          color: isActive ? 'red' : 'black',
          fontWeight: isActive ? 'bold' : 'normal'
        })}
      >
        About
      </NavLink>
      
      {/* Using end prop for exact matching */}
      <NavLink 
        to="/dashboard"
        className="nav-link"
        end // Only active when exactly "/dashboard"
      >
        Dashboard
      </NavLink>
    </nav>
  );
}

// CSS for active links
// .nav-link.active {
//   color: #007bff;
//   font-weight: bold;
// }
```

### Programmatic Navigation
```javascript
import { useNavigate, useLocation } from 'react-router-dom';

function LoginForm() {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Get the page user was trying to access
  const from = location.state?.from?.pathname || '/dashboard';

  const handleLogin = async (credentials) => {
    try {
      await login(credentials);
      
      // Navigate to intended page or dashboard
      navigate(from, { replace: true });
      
      // Or navigate with state
      navigate('/dashboard', { 
        state: { message: 'Welcome back!' } 
      });
      
    } catch (error) {
      console.error('Login failed:', error);
    }
  };

  const handleCancel = () => {
    // Go back to previous page
    navigate(-1);
    
    // Or go back specific number of pages
    navigate(-2);
    
    // Or go to specific page
    navigate('/');
  };

  return (
    <form onSubmit={handleLogin}>
      {/* Form fields */}
      <button type="submit">Login</button>
      <button type="button" onClick={handleCancel}>
        Cancel
      </button>
    </form>
  );
}

// Navigation with confirmation
function useNavigateWithConfirm() {
  const navigate = useNavigate();
  
  const navigateWithConfirm = (to, message = 'Are you sure?') => {
    if (window.confirm(message)) {
      navigate(to);
    }
  };
  
  return navigateWithConfirm;
}

function DangerousAction() {
  const navigateWithConfirm = useNavigateWithConfirm();
  
  const handleDelete = () => {
    // Perform delete operation
    navigateWithConfirm('/', 'Item deleted. Go to home?');
  };
  
  return <button onClick={handleDelete}>Delete</button>;
}
```

### Location and History
```javascript
import { useLocation, useNavigate } from 'react-router-dom';

function LocationExample() {
  const location = useLocation();
  const navigate = useNavigate();
  
  console.log('Current location:', location);
  // {
  //   pathname: '/users/123',
  //   search: '?tab=profile',
  //   hash: '#section1',
  //   state: { from: 'dashboard' },
  //   key: 'ac3df4'
  // }

  return (
    <div>
      <p>Current Path: {location.pathname}</p>
      <p>Search Params: {location.search}</p>
      <p>Hash: {location.hash}</p>
      
      {location.state && (
        <p>Came from: {location.state.from}</p>
      )}
      
      <button onClick={() => navigate(-1)}>
        Go Back
      </button>
    </div>
  );
}
```

## Route Guards and Protection

### Protected Routes
```javascript
import { Navigate, useLocation } from 'react-router-dom';

// Protected Route component
function ProtectedRoute({ children }) {
  const { user, loading } = useAuth(); // Custom auth hook
  const location = useLocation();

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!user) {
    // Redirect to login page with return url
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
}

// Role-based protection
function RoleProtectedRoute({ children, requiredRole }) {
  const { user } = useAuth();
  
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  
  if (user.role !== requiredRole) {
    return <Navigate to="/unauthorized" replace />;
  }
  
  return children;
}

// Usage in routes
function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/login" element={<Login />} />
      
      {/* Protected routes */}
      <Route path="/dashboard" element={
        <ProtectedRoute>
          <Dashboard />
        </ProtectedRoute>
      } />
      
      {/* Admin only routes */}
      <Route path="/admin" element={
        <RoleProtectedRoute requiredRole="admin">
          <AdminPanel />
        </RoleProtectedRoute>
      } />
      
      {/* Multiple protection levels */}
      <Route path="/settings" element={
        <ProtectedRoute>
          <SettingsLayout />
        </ProtectedRoute>
      }>
        <Route index element={<GeneralSettings />} />
        <Route path="billing" element={
          <RoleProtectedRoute requiredRole="owner">
            <BillingSettings />
          </RoleProtectedRoute>
        } />
      </Route>
    </Routes>
  );
}
```

### Authentication Hook
```javascript
import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on app start
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

  const login = async (credentials) => {
    const userData = await authenticateUser(credentials);
    setUser(userData);
    return userData;
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('token');
  };

  const value = {
    user,
    login,
    logout,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Helper functions
async function checkAuthStatus() {
  const token = localStorage.getItem('token');
  if (!token) throw new Error('No token');
  
  const response = await fetch('/api/me', {
    headers: { Authorization: `Bearer ${token}` }
  });
  
  if (!response.ok) throw new Error('Invalid token');
  
  return response.json();
}

async function authenticateUser(credentials) {
  const response = await fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials)
  });
  
  if (!response.ok) throw new Error('Authentication failed');
  
  const data = await response.json();
  localStorage.setItem('token', data.token);
  return data.user;
}
```

### Route-Level Guards
```javascript
// Custom hook for route guards
function useRouteGuard(condition, redirectTo = '/login') {
  const navigate = useNavigate();
  
  useEffect(() => {
    if (!condition) {
      navigate(redirectTo, { replace: true });
    }
  }, [condition, redirectTo, navigate]);
}

// Usage in components
function AdminDashboard() {
  const { user } = useAuth();
  
  // Redirect if not admin
  useRouteGuard(user?.role === 'admin', '/unauthorized');
  
  return (
    <div>
      <h1>Admin Dashboard</h1>
      {/* Admin content */}
    </div>
  );
}

function BillingPage() {
  const { user } = useAuth();
  
  // Multiple conditions
  useRouteGuard(
    user && user.subscription && user.subscription.active,
    '/upgrade'
  );
  
  return (
    <div>
      <h1>Billing</h1>
      {/* Billing content */}
    </div>
  );
}
```

## Advanced Routing Patterns

### Route Configuration
```javascript
// Centralized route configuration
const routes = [
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'about', element: <About /> },
      {
        path: 'dashboard',
        element: <ProtectedRoute><DashboardLayout /></ProtectedRoute>,
        children: [
          { index: true, element: <DashboardHome /> },
          { path: 'analytics', element: <Analytics /> },
          { path: 'users', element: <Users /> }
        ]
      }
    ]
  }
];

// Using useRoutes hook
function App() {
  const routing = useRoutes(routes);
  return routing;
}

// Or with createBrowserRouter (React Router v6.4+)
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    errorElement: <ErrorPage />,
    children: [
      { index: true, element: <Home /> },
      {
        path: 'dashboard',
        element: <Dashboard />,
        loader: dashboardLoader, // Data loading
        children: [
          { path: 'analytics', element: <Analytics /> }
        ]
      }
    ]
  }
]);

function App() {
  return <RouterProvider router={router} />;
}
```

### Dynamic Route Generation
```javascript
// Generate routes from configuration
const routeConfig = [
  { path: '/', component: 'Home', exact: true },
  { path: '/about', component: 'About' },
  { path: '/users/:id', component: 'UserProfile' },
  { path: '/admin', component: 'Admin', protected: true, role: 'admin' }
];

const componentMap = {
  Home: () => import('./components/Home'),
  About: () => import('./components/About'),
  UserProfile: () => import('./components/UserProfile'),
  Admin: () => import('./components/Admin')
};

function generateRoutes(config) {
  return config.map(route => {
    const Component = React.lazy(componentMap[route.component]);
    
    let element = (
      <Suspense fallback={<div>Loading...</div>}>
        <Component />
      </Suspense>
    );
    
    if (route.protected) {
      element = (
        <ProtectedRoute requiredRole={route.role}>
          {element}
        </ProtectedRoute>
      );
    }
    
    return (
      <Route 
        key={route.path}
        path={route.path} 
        element={element}
        index={route.exact}
      />
    );
  });
}

function App() {
  return (
    <Routes>
      {generateRoutes(routeConfig)}
    </Routes>
  );
}
```

### Route Middleware Pattern
```javascript
// Route middleware system
function createRouteMiddleware(middlewares) {
  return function RouteMiddleware({ children }) {
    const [canRender, setCanRender] = useState(false);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();
    
    useEffect(() => {
      const runMiddlewares = async () => {
        try {
          for (const middleware of middlewares) {
            const result = await middleware();
            if (!result.success) {
              navigate(result.redirectTo);
              return;
            }
          }
          setCanRender(true);
        } catch (error) {
          navigate('/error');
        } finally {
          setLoading(false);
        }
      };
      
      runMiddlewares();
    }, [navigate]);
    
    if (loading) return <div>Loading...</div>;
    if (!canRender) return null;
    
    return children;
  };
}

// Middleware functions
const authMiddleware = () => {
  const { user } = useAuth();
  return {
    success: !!user,
    redirectTo: '/login'
  };
};

const adminMiddleware = () => {
  const { user } = useAuth();
  return {
    success: user?.role === 'admin',
    redirectTo: '/unauthorized'
  };
};

const subscriptionMiddleware = () => {
  const { user } = useAuth();
  return {
    success: user?.subscription?.active,
    redirectTo: '/upgrade'
  };
};

// Usage
const AdminRoute = createRouteMiddleware([authMiddleware, adminMiddleware]);
const PremiumRoute = createRouteMiddleware([authMiddleware, subscriptionMiddleware]);

function App() {
  return (
    <Routes>
      <Route path="/admin" element={
        <AdminRoute>
          <AdminDashboard />
        </AdminRoute>
      } />
      
      <Route path="/premium" element={
        <PremiumRoute>
          <PremiumFeatures />
        </PremiumRoute>
      } />
    </Routes>
  );
}
```

## Data Loading

### Loader Functions (React Router v6.4+)
```javascript
import { createBrowserRouter, useLoaderData } from 'react-router-dom';

// Loader functions
async function userLoader({ params }) {
  const response = await fetch(`/api/users/${params.userId}`);
  if (!response.ok) {
    throw new Response('User not found', { status: 404 });
  }
  return response.json();
}

async function dashboardLoader() {
  const [users, posts, analytics] = await Promise.all([
    fetch('/api/users').then(res => res.json()),
    fetch('/api/posts').then(res => res.json()),
    fetch('/api/analytics').then(res => res.json())
  ]);
  
  return { users, posts, analytics };
}

// Router with loaders
const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        path: 'users/:userId',
        element: <UserProfile />,
        loader: userLoader
      },
      {
        path: 'dashboard',
        element: <Dashboard />,
        loader: dashboardLoader
      }
    ]
  }
]);

// Components using loaded data
function UserProfile() {
  const user = useLoaderData();
  
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}

function Dashboard() {
  const { users, posts, analytics } = useLoaderData();
  
  return (
    <div>
      <h1>Dashboard</h1>
      <div>Total Users: {users.length}</div>
      <div>Total Posts: {posts.length}</div>
      <div>Views: {analytics.views}</div>
    </div>
  );
}
```

### Data Fetching Patterns
```javascript
// Traditional data fetching in component
function UserProfile() {
  const { userId } = useParams();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchUser(userId)
      .then(setUser)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [userId]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!user) return <div>User not found</div>;

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}

// Using React Query for better data management
import { useQuery } from '@tanstack/react-query';

function UserProfile() {
  const { userId } = useParams();
  const { 
    data: user, 
    isLoading, 
    error 
  } = useQuery({
    queryKey: ['user', userId],
    queryFn: () => fetchUser(userId),
    enabled: !!userId
  });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!user) return <div>User not found</div>;

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}
```

## Error Handling

### Error Boundaries for Routes
```javascript
import { ErrorBoundary } from 'react-error-boundary';

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div className="error-container">
      <h2>Something went wrong</h2>
      <p>{error.message}</p>
      <button onClick={resetErrorBoundary}>Try again</button>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          
          <Route path="dashboard" element={
            <ErrorBoundary 
              FallbackComponent={ErrorFallback}
              onReset={() => window.location.reload()}
            >
              <Dashboard />
            </ErrorBoundary>
          } />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```

### Error Pages
```javascript
function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="about" element={<About />} />
        
        {/* Error routes */}
        <Route path="unauthorized" element={<UnauthorizedPage />} />
        <Route path="server-error" element={<ServerErrorPage />} />
        
        {/* 404 - Must be last */}
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

function NotFoundPage() {
  return (
    <div className="error-page">
      <h1>404 - Page Not Found</h1>
      <p>The page you're looking for doesn't exist.</p>
      <Link to="/">Go back to Home</Link>
    </div>
  );
}

function UnauthorizedPage() {
  return (
    <div className="error-page">
      <h1>401 - Unauthorized</h1>
      <p>You don't have permission to access this page.</p>
      <Link to="/login">Login</Link>
    </div>
  );
}
```

### Global Error Handling
```javascript
// Error handling hook
function useErrorHandler() {
  const navigate = useNavigate();
  
  return useCallback((error) => {
    console.error('Application error:', error);
    
    if (error.status === 401) {
      navigate('/login');
    } else if (error.status === 403) {
      navigate('/unauthorized');
    } else if (error.status >= 500) {
      navigate('/server-error');
    } else {
      // Show toast or modal
      showErrorToast(error.message);
    }
  }, [navigate]);
}

// Usage in components
function DataComponent() {
  const handleError = useErrorHandler();
  
  const fetchData = async () => {
    try {
      const data = await apiCall();
      setData(data);
    } catch (error) {
      handleError(error);
    }
  };
  
  return (
    <div>
      <button onClick={fetchData}>Fetch Data</button>
    </div>
  );
}
```

---

*Continue to: [05-react-state-management.md](./05-react-state-management.md)*
