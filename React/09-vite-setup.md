# Vite Setup and Configuration

## Table of Contents
- [What is Vite?](#what-is-vite)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Asset Handling](#asset-handling)
- [CSS and Preprocessing](#css-and-preprocessing)
- [TypeScript Setup](#typescript-setup)
- [Plugins](#plugins)
- [Build and Deployment](#build-and-deployment)
- [Development Features](#development-features)
- [Performance Optimization](#performance-optimization)

## What is Vite?

Vite (French for "quick") is a modern build tool that provides:
- **Lightning fast cold server start**
- **Instant Hot Module Replacement (HMR)**
- **Optimized builds** using Rollup
- **Rich plugin ecosystem**
- **TypeScript support** out of the box

### Vite vs Create React App

| Feature | Vite | Create React App |
|---------|------|------------------|
| **Dev server startup** | Instant | Slower |
| **HMR** | Instant | Fast |
| **Bundle size** | Smaller | Larger |
| **Configuration** | Flexible | Limited |
| **TypeScript** | Built-in | Needs setup |
| **Plugins** | Rich ecosystem | Limited |

## Getting Started

### Creating a New Project
```bash
# Create React project with Vite
npm create vite@latest my-react-app -- --template react

# With TypeScript
npm create vite@latest my-react-app -- --template react-ts

# Navigate to project
cd my-react-app

# Install dependencies
npm install

# Start development server
npm run dev
```

### Available Templates
```bash
# JavaScript templates
npm create vite@latest my-app -- --template vanilla
npm create vite@latest my-app -- --template react
npm create vite@latest my-app -- --template vue
npm create vite@latest my-app -- --template svelte

# TypeScript templates
npm create vite@latest my-app -- --template vanilla-ts
npm create vite@latest my-app -- --template react-ts
npm create vite@latest my-app -- --template vue-ts
npm create vite@latest my-app -- --template svelte-ts
```

### Manual Setup
```bash
# Create directory
mkdir my-react-app
cd my-react-app

# Initialize package.json
npm init -y

# Install Vite and React
npm install vite @vitejs/plugin-react react react-dom

# Install dev dependencies
npm install -D @types/react @types/react-dom typescript
```

## Project Structure

### Basic Vite + React Structure
```
my-react-app/
├── public/
│   ├── vite.svg
│   └── favicon.ico
├── src/
│   ├── assets/
│   │   └── react.svg
│   ├── components/
│   │   └── Header.jsx
│   ├── pages/
│   │   ├── Home.jsx
│   │   └── About.jsx
│   ├── hooks/
│   │   └── useLocalStorage.js
│   ├── utils/
│   │   └── helpers.js
│   ├── styles/
│   │   ├── global.css
│   │   └── components.css
│   ├── App.jsx
│   ├── App.css
│   ├── main.jsx
│   └── index.css
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

### Key Files

#### index.html (Entry Point)
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vite + React</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

#### main.jsx (App Entry)
```javascript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

#### package.json Scripts
```json
{
  "name": "my-react-app",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint . --ext js,jsx,ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "eslint": "^8.55.0",
    "eslint-plugin-react": "^7.33.2",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }
}
```

## Configuration

### Basic vite.config.js
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  
  // Development server configuration
  server: {
    port: 3000,
    open: true, // automatically open browser
    host: true, // expose to network
  },
  
  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'terser',
  },
  
  // Path resolution
  resolve: {
    alias: {
      '@': '/src',
      '@components': '/src/components',
      '@pages': '/src/pages',
      '@utils': '/src/utils',
      '@assets': '/src/assets',
    }
  }
})
```

### Advanced Configuration
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    react({
      // Enable React Fast Refresh
      fastRefresh: true,
      
      // JSX runtime configuration
      jsxRuntime: 'automatic',
      
      // Babel configuration
      babel: {
        plugins: [
          // Add any Babel plugins here
        ]
      }
    })
  ],
  
  // Development server
  server: {
    port: 3000,
    strictPort: true,
    host: true,
    open: true,
    cors: true,
    
    // Proxy API requests
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    },
    
    // Custom headers
    headers: {
      'Access-Control-Allow-Origin': '*',
    }
  },
  
  // Preview server (for built app)
  preview: {
    port: 4173,
    strictPort: true,
    host: true,
    open: true
  },
  
  // Build options
  build: {
    target: 'esnext',
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: 'terser',
    
    // Rollup options
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        // Add multiple entry points if needed
      },
      
      output: {
        manualChunks: {
          // Split vendor chunks
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
        }
      }
    },
    
    // Terser options
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      }
    },
    
    // Asset file size warning limit
    chunkSizeWarningLimit: 1000,
  },
  
  // Path aliases
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@pages': resolve(__dirname, 'src/pages'),
      '@hooks': resolve(__dirname, 'src/hooks'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@assets': resolve(__dirname, 'src/assets'),
      '@styles': resolve(__dirname, 'src/styles'),
    }
  },
  
  // CSS configuration
  css: {
    devSourcemap: true,
    modules: {
      localsConvention: 'camelCase'
    },
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/variables.scss";`
      }
    }
  },
  
  // Define global constants
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
    __API_URL__: JSON.stringify(process.env.VITE_API_URL || 'http://localhost:8000'),
  },
  
  // Optimize dependencies
  optimizeDeps: {
    include: ['react', 'react-dom'],
    exclude: ['some-package']
  }
})
```

### Environment-Based Configuration
```javascript
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command, mode }) => {
  // Load env file based on `mode` in the current working directory
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [react()],
    
    server: {
      port: parseInt(env.VITE_PORT) || 3000,
    },
    
    build: {
      sourcemap: mode === 'development',
      minify: mode === 'production' ? 'terser' : false,
    },
    
    define: {
      __DEV__: mode === 'development',
      __PROD__: mode === 'production',
      'process.env.NODE_ENV': JSON.stringify(mode),
    }
  }
})
```

## Environment Variables

### .env Files
```bash
# .env (default for all environments)
VITE_APP_TITLE=My React App
VITE_API_BASE_URL=https://api.example.com

# .env.local (local overrides, gitignored)
VITE_API_BASE_URL=http://localhost:8000

# .env.development (development mode)
VITE_LOG_LEVEL=debug
VITE_ENABLE_DEVTOOLS=true

# .env.production (production mode)
VITE_LOG_LEVEL=error
VITE_ENABLE_DEVTOOLS=false

# .env.staging (staging environment)
VITE_API_BASE_URL=https://staging-api.example.com
```

### Using Environment Variables
```javascript
// In your React components
function App() {
  const apiUrl = import.meta.env.VITE_API_BASE_URL;
  const appTitle = import.meta.env.VITE_APP_TITLE;
  const isDev = import.meta.env.DEV;
  const isProd = import.meta.env.PROD;
  const mode = import.meta.env.MODE;
  
  return (
    <div>
      <h1>{appTitle}</h1>
      <p>API URL: {apiUrl}</p>
      <p>Mode: {mode}</p>
      {isDev && <div>Development mode</div>}
    </div>
  );
}

// Type definitions for TypeScript
// vite-env.d.ts
interface ImportMetaEnv {
  readonly VITE_APP_TITLE: string;
  readonly VITE_API_BASE_URL: string;
  readonly VITE_LOG_LEVEL: string;
  readonly VITE_ENABLE_DEVTOOLS: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

### Environment Configuration
```javascript
// config/env.js
const config = {
  development: {
    apiUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    logLevel: 'debug',
    enableDevTools: true,
  },
  
  production: {
    apiUrl: import.meta.env.VITE_API_BASE_URL || 'https://api.example.com',
    logLevel: 'error',
    enableDevTools: false,
  },
  
  staging: {
    apiUrl: import.meta.env.VITE_API_BASE_URL || 'https://staging-api.example.com',
    logLevel: 'warn',
    enableDevTools: true,
  }
};

const currentConfig = config[import.meta.env.MODE] || config.development;

export default currentConfig;

// Usage
import config from '@/config/env';

fetch(`${config.apiUrl}/users`)
  .then(response => response.json())
  .then(data => console.log(data));
```

## Asset Handling

### Static Assets
```javascript
// Importing assets
import reactLogo from '@/assets/react.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      {/* Static asset from public folder */}
      <img src="/logo.png" alt="Logo" />
      
      {/* Imported asset (gets hashed filename) */}
      <img src={reactLogo} alt="React Logo" />
      
      {/* Dynamic import */}
      <img src={new URL('./assets/dynamic.png', import.meta.url).href} alt="Dynamic" />
    </div>
  );
}
```

### Asset Organization
```
src/
├── assets/
│   ├── images/
│   │   ├── logo.svg
│   │   ├── hero-bg.jpg
│   │   └── icons/
│   │       ├── user.svg
│   │       └── settings.svg
│   ├── fonts/
│   │   ├── roboto.woff2
│   │   └── inter.woff2
│   ├── videos/
│   │   └── intro.mp4
│   └── data/
│       └── countries.json
```

### Asset Processing
```javascript
// vite.config.js
export default defineConfig({
  assetsInclude: ['**/*.gltf'], // Include additional file types
  
  build: {
    assetsInlineLimit: 4096, // Files smaller than 4kb will be inlined
    rollupOptions: {
      output: {
        assetFileNames: (assetInfo) => {
          let extType = assetInfo.name.split('.').at(1);
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(extType)) {
            extType = 'img';
          }
          return `assets/${extType}/[name]-[hash][extname]`;
        },
        chunkFileNames: 'assets/js/[name]-[hash].js',
        entryFileNames: 'assets/js/[name]-[hash].js',
      }
    }
  }
});

// Using assets in CSS
.hero {
  background-image: url('@/assets/images/hero-bg.jpg');
}

// Importing JSON data
import countriesData from '@/assets/data/countries.json';

function CountrySelect() {
  return (
    <select>
      {countriesData.map(country => (
        <option key={country.code} value={country.code}>
          {country.name}
        </option>
      ))}
    </select>
  );
}
```

## CSS and Preprocessing

### CSS Modules
```css
/* Button.module.css */
.button {
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  background-color: var(--primary-color);
  color: white;
  cursor: pointer;
}

.button:hover {
  background-color: var(--primary-color-dark);
}

.primary {
  background-color: #007bff;
}

.secondary {
  background-color: #6c757d;
}
```

```javascript
// Button.jsx
import styles from './Button.module.css';

function Button({ children, variant = 'primary', ...props }) {
  return (
    <button 
      className={`${styles.button} ${styles[variant]}`}
      {...props}
    >
      {children}
    </button>
  );
}
```

### Sass/SCSS Setup
```bash
# Install Sass
npm install -D sass
```

```scss
// styles/variables.scss
$primary-color: #007bff;
$secondary-color: #6c757d;
$font-family: 'Inter', sans-serif;

$breakpoints: (
  mobile: 768px,
  tablet: 1024px,
  desktop: 1200px,
);

@mixin mobile {
  @media (max-width: map-get($breakpoints, mobile)) {
    @content;
  }
}

@mixin tablet {
  @media (max-width: map-get($breakpoints, tablet)) {
    @content;
  }
}
```

```scss
// components/Button.scss
@import '@/styles/variables';

.button {
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  background-color: $primary-color;
  font-family: $font-family;
  cursor: pointer;
  
  @include mobile {
    padding: 8px 16px;
    font-size: 14px;
  }
  
  &:hover {
    background-color: darken($primary-color, 10%);
  }
  
  &.secondary {
    background-color: $secondary-color;
  }
}
```

### CSS-in-JS with Styled Components
```bash
npm install styled-components
npm install -D @types/styled-components
```

```javascript
// components/StyledButton.jsx
import styled from 'styled-components';

const StyledButton = styled.button`
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  background-color: ${props => props.variant === 'primary' ? '#007bff' : '#6c757d'};
  color: white;
  font-size: 16px;
  cursor: pointer;
  transition: background-color 0.2s ease;
  
  &:hover {
    background-color: ${props => props.variant === 'primary' ? '#0056b3' : '#545b62'};
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  
  @media (max-width: 768px) {
    padding: 8px 16px;
    font-size: 14px;
  }
`;

function Button({ children, variant = 'primary', ...props }) {
  return (
    <StyledButton variant={variant} {...props}>
      {children}
    </StyledButton>
  );
}
```

### PostCSS Configuration
```bash
npm install -D postcss autoprefixer
```

```javascript
// postcss.config.js
export default {
  plugins: {
    autoprefixer: {},
    'postcss-nested': {},
    'postcss-custom-media': {},
  }
}
```

## TypeScript Setup

### Basic TypeScript Configuration
```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@/components/*": ["./src/components/*"],
      "@/pages/*": ["./src/pages/*"],
      "@/hooks/*": ["./src/hooks/*"],
      "@/utils/*": ["./src/utils/*"],
      "@/types/*": ["./src/types/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

```json
// tsconfig.node.json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

### TypeScript Vite Config
```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@/components': resolve(__dirname, './src/components'),
      '@/pages': resolve(__dirname, './src/pages'),
      '@/hooks': resolve(__dirname, './src/hooks'),
      '@/utils': resolve(__dirname, './src/utils'),
      '@/types': resolve(__dirname, './src/types'),
    }
  }
})
```

### Type Definitions
```typescript
// src/types/index.ts
export interface User {
  id: number;
  name: string;
  email: string;
  avatar?: string;
  role: 'admin' | 'user';
  createdAt: string;
}

export interface ApiResponse<T> {
  data: T;
  message: string;
  success: boolean;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// Component prop types
export interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
}
```

## Plugins

### Essential Plugins
```bash
# React plugin (included by default)
npm install -D @vitejs/plugin-react

# ESLint integration
npm install -D vite-plugin-eslint

# Bundle analyzer
npm install -D rollup-plugin-visualizer

# PWA
npm install -D vite-plugin-pwa

# Environment variables in HTML
npm install -D vite-plugin-html
```

### Plugin Configuration
```javascript
// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import eslint from 'vite-plugin-eslint'
import { visualizer } from 'rollup-plugin-visualizer'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    
    // ESLint integration
    eslint({
      cache: false,
      include: ['./src/**/*.js', './src/**/*.jsx', './src/**/*.ts', './src/**/*.tsx'],
      exclude: [],
    }),
    
    // Bundle analyzer
    visualizer({
      filename: 'dist/stats.html',
      open: true,
      gzipSize: true,
    }),
    
    // PWA
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}']
      },
      manifest: {
        name: 'My React App',
        short_name: 'ReactApp',
        description: 'My awesome React application',
        theme_color: '#ffffff',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          }
        ]
      }
    })
  ]
})
```

### Custom Plugin Example
```javascript
// plugins/example-plugin.js
function examplePlugin() {
  return {
    name: 'example-plugin',
    configResolved(config) {
      console.log('Config resolved:', config.command);
    },
    buildStart() {
      console.log('Build started');
    },
    transformIndexHtml(html) {
      return html.replace('{{TITLE}}', 'My Awesome App');
    }
  };
}

// Usage in vite.config.js
import examplePlugin from './plugins/example-plugin';

export default defineConfig({
  plugins: [
    react(),
    examplePlugin()
  ]
});
```

## Build and Deployment

### Build Configuration
```javascript
// vite.config.js
export default defineConfig({
  build: {
    target: 'esnext',
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: 'terser',
    
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
})
```

### Build Scripts
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "build:staging": "vite build --mode staging",
    "build:analyze": "vite build && npx vite-bundle-analyzer dist/stats.html",
    "preview": "vite preview",
    "preview:network": "vite preview --host",
    "clean": "rm -rf dist"
  }
}
```

### Deployment Examples

#### Netlify
```bash
# netlify.toml
[build]
  publish = "dist"
  command = "npm run build"

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

#### Vercel
```json
// vercel.json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "dist"
      }
    }
  ],
  "routes": [
    {
      "handle": "filesystem"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

#### GitHub Pages
```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout
      uses: actions/checkout@v2

    - name: Setup Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '18'

    - name: Install dependencies
      run: npm ci

    - name: Build
      run: npm run build

    - name: Deploy
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./dist
```

## Development Features

### Hot Module Replacement (HMR)
```javascript
// Automatic HMR for React components
// No configuration needed with @vitejs/plugin-react

// Manual HMR for other modules
if (import.meta.hot) {
  import.meta.hot.accept('./dependency.js', (newModule) => {
    // Handle module update
  });
  
  import.meta.hot.dispose((data) => {
    // Cleanup before module is replaced
  });
}
```

### Development Tools
```javascript
// vite.config.js
export default defineConfig({
  server: {
    open: true, // Auto-open browser
    host: true, // Expose to network
    
    // API proxy
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
```

### Mock Data in Development
```javascript
// src/mocks/api.js
const mockUsers = [
  { id: 1, name: 'John Doe', email: 'john@example.com' },
  { id: 2, name: 'Jane Smith', email: 'jane@example.com' },
];

export const mockApi = {
  getUsers: () => Promise.resolve(mockUsers),
  getUser: (id) => Promise.resolve(mockUsers.find(u => u.id === id)),
};

// src/api/users.js
import { mockApi } from '../mocks/api';

const isDev = import.meta.env.DEV;
const useMocks = import.meta.env.VITE_USE_MOCKS === 'true';

export const userApi = {
  async getUsers() {
    if (isDev && useMocks) {
      return mockApi.getUsers();
    }
    
    const response = await fetch('/api/users');
    return response.json();
  }
};
```

## Performance Optimization

### Code Splitting
```javascript
// Lazy loading components
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Profile = lazy(() => import('./pages/Profile'));

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={
          <Suspense fallback={<div>Loading...</div>}>
            <Dashboard />
          </Suspense>
        } />
        <Route path="/profile" element={
          <Suspense fallback={<div>Loading...</div>}>
            <Profile />
          </Suspense>
        } />
      </Routes>
    </Router>
  );
}

// Dynamic imports
const loadFeature = async () => {
  const module = await import('./features/AdvancedFeature');
  return module.default;
};
```

### Bundle Optimization
```javascript
// vite.config.js
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Vendor chunks
          if (id.includes('node_modules')) {
            if (id.includes('react')) {
              return 'react-vendor';
            }
            if (id.includes('lodash')) {
              return 'lodash';
            }
            return 'vendor';
          }
          
          // Feature-based chunks
          if (id.includes('/src/features/dashboard')) {
            return 'dashboard';
          }
          if (id.includes('/src/features/profile')) {
            return 'profile';
          }
        }
      }
    }
  },
  
  // Optimize dependencies
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom'
    ],
    exclude: ['@vite/client', '@vite/env']
  }
})
```

### Asset Optimization
```javascript
// vite.config.js
export default defineConfig({
  build: {
    assetsInlineLimit: 8192, // 8kb threshold for inlining
    
    rollupOptions: {
      output: {
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name.split('.');
          const ext = info[info.length - 1];
          
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext)) {
            return `assets/images/[name]-[hash][extname]`;
          }
          
          if (/woff2?|eot|ttf|otf/i.test(ext)) {
            return `assets/fonts/[name]-[hash][extname]`;
          }
          
          return `assets/[name]-[hash][extname]`;
        }
      }
    }
  }
})
```

---

*Continue to: [10-react-native-basics.md](./10-react-native-basics.md)*
