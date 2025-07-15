# React Basics

## Table of Contents
- [What is React?](#what-is-react)
- [Why React?](#why-react)
- [React Core Concepts](#react-core-concepts)
- [JSX](#jsx)
- [Virtual DOM](#virtual-dom)
- [Creating Your First React App](#creating-your-first-react-app)
- [React Developer Tools](#react-developer-tools)
- [Basic Project Structure](#basic-project-structure)

## What is React?

React is a **JavaScript library** for building user interfaces, particularly web applications. Created by Facebook (Meta) in 2013, React focuses on creating reusable UI components and managing application state efficiently.

### Key Characteristics:
- **Component-Based**: Build encapsulated components that manage their own state
- **Declarative**: Describe what the UI should look like for any given state
- **Learn Once, Write Anywhere**: Can be used for web, mobile, and desktop applications

## Why React?

### Advantages:
1. **Virtual DOM**: Efficient updates and rendering
2. **Component Reusability**: Write once, use anywhere
3. **Large Ecosystem**: Extensive third-party libraries
4. **Strong Community**: Great support and resources
5. **Performance**: Optimized rendering with reconciliation
6. **Developer Experience**: Great tooling and debugging

### Disadvantages:
1. **Learning Curve**: JSX and concepts can be initially confusing
2. **Rapid Changes**: Frequent updates and new patterns
3. **Boilerplate**: Can require additional setup for complex apps

## React Core Concepts

### 1. Components
Components are the building blocks of React applications. They can be:

**Function Components (Recommended)**:
```javascript
function Welcome(props) {
  return <h1>Hello, {props.name}!</h1>;
}

// Arrow function syntax
const Welcome = (props) => {
  return <h1>Hello, {props.name}!</h1>;
};
```

**Class Components (Legacy)**:
```javascript
class Welcome extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}!</h1>;
  }
}
```

### 2. Props (Properties)
Props are inputs to components - data passed from parent to child components.

```javascript
// Parent component
function App() {
  return (
    <div>
      <Welcome name="Alice" age={25} />
      <Welcome name="Bob" age={30} />
    </div>
  );
}

// Child component
function Welcome({ name, age }) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      <p>You are {age} years old.</p>
    </div>
  );
}
```

### 3. State
State is internal component data that can change over time.

```javascript
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        Increment
      </button>
    </div>
  );
}
```

## JSX

JSX (JavaScript XML) is a syntax extension that allows you to write HTML-like code in JavaScript.

### JSX Rules:
1. **Return single element** or Fragment
2. **Close all tags** (including self-closing)
3. **Use camelCase** for attributes
4. **className instead of class**

```javascript
// ✅ Good JSX
function MyComponent() {
  return (
    <div className="container">
      <h1>Welcome</h1>
      <img src="image.jpg" alt="Description" />
      <input type="text" />
    </div>
  );
}

// ✅ Using Fragment
function MyComponent() {
  return (
    <>
      <h1>Title</h1>
      <p>Paragraph</p>
    </>
  );
}

// ❌ Multiple elements without wrapper
function MyComponent() {
  return (
    <h1>Title</h1>
    <p>Paragraph</p>
  );
}
```

### JSX Expressions
You can embed JavaScript expressions in JSX using curly braces:

```javascript
function UserProfile({ user }) {
  const isLoggedIn = user !== null;
  
  return (
    <div>
      <h1>{user ? `Welcome, ${user.name}!` : 'Please log in'}</h1>
      {isLoggedIn && <p>Last login: {user.lastLogin}</p>}
      {isLoggedIn ? (
        <button>Logout</button>
      ) : (
        <button>Login</button>
      )}
    </div>
  );
}
```

### Conditional Rendering
```javascript
function WelcomeMessage({ isLoggedIn, userName }) {
  // Method 1: Ternary operator
  return (
    <div>
      {isLoggedIn ? (
        <h1>Welcome back, {userName}!</h1>
      ) : (
        <h1>Please sign in.</h1>
      )}
    </div>
  );
}

// Method 2: Logical AND
function Notification({ hasNewMessages, messageCount }) {
  return (
    <div>
      {hasNewMessages && (
        <p>You have {messageCount} new messages!</p>
      )}
    </div>
  );
}

// Method 3: if/else before return
function Dashboard({ user }) {
  if (!user) {
    return <div>Loading...</div>;
  }
  
  if (user.role !== 'admin') {
    return <div>Access denied</div>;
  }
  
  return <div>Admin Dashboard</div>;
}
```

### Lists and Keys
```javascript
function TodoList({ todos }) {
  return (
    <ul>
      {todos.map((todo) => (
        <li key={todo.id}>
          <span>{todo.text}</span>
          <button>Delete</button>
        </li>
      ))}
    </ul>
  );
}

// Key importance example
const numbers = [1, 2, 3, 4, 5];
const listItems = numbers.map((number) =>
  <li key={number.toString()}>
    {number}
  </li>
);
```

## Virtual DOM

The Virtual DOM is a JavaScript representation of the real DOM kept in memory and synced with the "real" DOM.

### How it works:
1. **State Change**: Component state changes
2. **Virtual DOM Update**: New virtual DOM tree is created
3. **Diffing**: React compares (diffs) the new tree with the previous tree
4. **Reconciliation**: React updates only the changed parts in the real DOM

### Benefits:
- **Performance**: Minimizes expensive DOM operations
- **Predictability**: Makes UI updates more predictable
- **Abstraction**: Developers don't need to manually manipulate DOM

```javascript
// When state changes, React efficiently updates only what's needed
function App() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('');

  return (
    <div>
      <h1>Count: {count}</h1> {/* Only this updates when count changes */}
      <button onClick={() => setCount(count + 1)}>+</button>
      
      <h2>Name: {name}</h2> {/* Only this updates when name changes */}
      <input 
        value={name} 
        onChange={(e) => setName(e.target.value)} 
      />
    </div>
  );
}
```

## Creating Your First React App

### Using Vite (Recommended)
```bash
# Create new React app with Vite
npm create vite@latest my-react-app -- --template react

# With TypeScript
npm create vite@latest my-react-app -- --template react-ts

# Navigate and install dependencies
cd my-react-app
npm install

# Start development server
npm run dev
```

### Using Create React App (CRA)
```bash
# Create new React app
npx create-react-app my-react-app

# With TypeScript
npx create-react-app my-react-app --template typescript

# Navigate and start
cd my-react-app
npm start
```

### Why Vite over CRA?
- **Faster**: Lightning-fast cold server start
- **HMR**: Instant hot module replacement
- **Modern**: Built for modern web development
- **Flexible**: Less opinionated, more configurable

## React Developer Tools

### Browser Extension
Install React Developer Tools for Chrome/Firefox:
- **Components Tab**: Inspect component hierarchy and props
- **Profiler Tab**: Analyze performance and re-renders

### Useful Features:
```javascript
// Add display names for debugging
function MyComponent() {
  return <div>Hello</div>;
}
MyComponent.displayName = 'MyComponent';

// Use React.StrictMode for development warnings
function App() {
  return (
    <React.StrictMode>
      <MyComponent />
    </React.StrictMode>
  );
}
```

## Basic Project Structure

```
my-react-app/
├── public/
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── components/
│   │   ├── Header.jsx
│   │   └── Footer.jsx
│   ├── pages/
│   │   ├── Home.jsx
│   │   └── About.jsx
│   ├── hooks/
│   │   └── useLocalStorage.js
│   ├── utils/
│   │   └── helpers.js
│   ├── styles/
│   │   └── global.css
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── package.json
└── vite.config.js
```

### File Naming Conventions:
- **Components**: PascalCase (`UserProfile.jsx`)
- **Hooks**: camelCase starting with "use" (`useAuth.js`)
- **Utilities**: camelCase (`formatDate.js`)
- **Constants**: UPPER_SNAKE_CASE (`API_ENDPOINTS.js`)

### Basic App.jsx Structure:
```javascript
import { useState } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import './App.css';

function App() {
  const [user, setUser] = useState(null);

  return (
    <div className="App">
      <Header user={user} />
      <main>
        <h1>Welcome to My React App</h1>
        {/* Your app content here */}
      </main>
      <Footer />
    </div>
  );
}

export default App;
```

### Environment Variables:
```javascript
// .env file
VITE_API_URL=https://api.example.com
VITE_APP_TITLE=My React App

// Usage in component
const apiUrl = import.meta.env.VITE_API_URL;
const appTitle = import.meta.env.VITE_APP_TITLE;
```

## Next Steps

After mastering these basics, move on to:
1. **Components and Props** (detailed component patterns)
2. **React Hooks** (useState, useEffect, custom hooks)
3. **Event Handling** (forms, user interactions)
4. **State Management** (Context API, external libraries)
5. **Routing** (React Router)

---

*Continue to: [02-react-components.md](./02-react-components.md)*
