# Shared Patterns Between React and React Native

## Table of Contents
- [Cross-Platform Code Organization](#cross-platform-code-organization)
- [Shared Business Logic](#shared-business-logic)
- [Common State Management](#common-state-management)
- [Universal Components](#universal-components)
- [Platform-Specific Implementations](#platform-specific-implementations)
- [Shared Utilities and Helpers](#shared-utilities-and-helpers)
- [Testing Strategies](#testing-strategies)
- [Build and Deployment](#build-and-deployment)

## Cross-Platform Code Organization

### Monorepo Structure
```
my-app/
├── packages/
│   ├── shared/                     # Shared business logic
│   │   ├── src/
│   │   │   ├── hooks/             # Custom hooks
│   │   │   ├── utils/             # Utility functions
│   │   │   ├── types/             # TypeScript types
│   │   │   ├── constants/         # Constants
│   │   │   ├── services/          # API services
│   │   │   └── validators/        # Validation schemas
│   │   ├── package.json
│   │   └── tsconfig.json
│   ├── mobile/                    # React Native app
│   │   ├── src/
│   │   │   ├── components/
│   │   │   ├── screens/
│   │   │   ├── navigation/
│   │   │   └── platform/          # Platform-specific code
│   │   ├── package.json
│   │   └── metro.config.js
│   └── web/                       # React web app
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   └── platform/          # Platform-specific code
│       ├── package.json
│       └── vite.config.ts
├── package.json                   # Root package.json
└── tsconfig.json                  # Root TypeScript config
```

### Package.json Configuration
```json
// Root package.json
{
  "name": "my-app-monorepo",
  "private": true,
  "workspaces": [
    "packages/*"
  ],
  "scripts": {
    "dev:web": "cd packages/web && npm run dev",
    "dev:mobile": "cd packages/mobile && npm run start",
    "build:web": "cd packages/web && npm run build",
    "build:mobile": "cd packages/mobile && npm run build",
    "test": "npm run test --workspaces",
    "lint": "npm run lint --workspaces",
    "type-check": "npm run type-check --workspaces"
  },
  "devDependencies": {
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "eslint": "^8.0.0",
    "typescript": "^5.0.0"
  }
}

// packages/shared/package.json
{
  "name": "@my-app/shared",
  "version": "1.0.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "watch": "tsc --watch",
    "test": "jest",
    "lint": "eslint src --ext .ts,.tsx"
  },
  "dependencies": {
    "zod": "^3.22.0",
    "date-fns": "^2.30.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/jest": "^29.0.0",
    "jest": "^29.0.0"
  }
}

// packages/mobile/package.json
{
  "name": "@my-app/mobile",
  "version": "1.0.0",
  "scripts": {
    "start": "expo start",
    "android": "expo start --android",
    "ios": "expo start --ios",
    "web": "expo start --web",
    "build": "eas build",
    "test": "jest"
  },
  "dependencies": {
    "@my-app/shared": "*",
    "react-native": "0.72.0",
    "expo": "~49.0.0"
  }
}

// packages/web/package.json
{
  "name": "@my-app/web",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "jest"
  },
  "dependencies": {
    "@my-app/shared": "*",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
```

### TypeScript Configuration
```json
// Root tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@my-app/shared": ["packages/shared/src"],
      "@my-app/shared/*": ["packages/shared/src/*"]
    },
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "references": [
    { "path": "./packages/shared" },
    { "path": "./packages/mobile" },
    { "path": "./packages/web" }
  ]
}

// packages/shared/tsconfig.json
{
  "extends": "../../tsconfig.json",
  "compilerOptions": {
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "composite": true
  },
  "include": ["src/**/*"],
  "exclude": ["dist", "node_modules"]
}
```

## Shared Business Logic

### API Service Layer
```typescript
// packages/shared/src/services/api.ts
import { z } from 'zod';

// Shared API configuration
export const API_CONFIG = {
  BASE_URL: process.env.API_BASE_URL || 'https://api.example.com',
  TIMEOUT: 10000,
  RETRY_ATTEMPTS: 3,
};

// Request/Response types
export const UserSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  name: z.string(),
  avatar: z.string().optional(),
  createdAt: z.string().datetime(),
});

export const ApiResponseSchema = z.object({
  success: z.boolean(),
  data: z.any().optional(),
  error: z.string().optional(),
  pagination: z.object({
    page: z.number(),
    limit: z.number(),
    total: z.number(),
  }).optional(),
});

export type User = z.infer<typeof UserSchema>;
export type ApiResponse<T = any> = z.infer<typeof ApiResponseSchema> & { data?: T };

// Base API client
export class ApiClient {
  private baseURL: string;
  private timeout: number;
  private headers: Record<string, string>;

  constructor(config: typeof API_CONFIG) {
    this.baseURL = config.BASE_URL;
    this.timeout = config.TIMEOUT;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  setAuthToken(token: string) {
    this.headers.Authorization = `Bearer ${token}`;
  }

  removeAuthToken() {
    delete this.headers.Authorization;
  }

  async request<T>(
    endpoint: string,
    options: {
      method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
      body?: any;
      headers?: Record<string, string>;
      timeout?: number;
    } = {}
  ): Promise<ApiResponse<T>> {
    const {
      method = 'GET',
      body,
      headers: customHeaders = {},
      timeout = this.timeout,
    } = options;

    const url = `${this.baseURL}${endpoint}`;
    const requestHeaders = { ...this.headers, ...customHeaders };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(url, {
        method,
        headers: requestHeaders,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      return ApiResponseSchema.parse(data) as ApiResponse<T>;
    } catch (error) {
      throw new Error(`API request failed: ${error.message}`);
    }
  }

  // CRUD operations
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<ApiResponse<T>> {
    const query = params ? `?${new URLSearchParams(params)}` : '';
    return this.request<T>(`${endpoint}${query}`);
  }

  async post<T>(endpoint: string, data: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'POST', body: data });
  }

  async put<T>(endpoint: string, data: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'PUT', body: data });
  }

  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }
}

// Service instances
export const apiClient = new ApiClient(API_CONFIG);

// User service
export const userService = {
  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>('/users/me');
    return UserSchema.parse(response.data);
  },

  async updateUser(id: string, updates: Partial<User>): Promise<User> {
    const response = await apiClient.put<User>(`/users/${id}`, updates);
    return UserSchema.parse(response.data);
  },

  async getUsers(page = 1, limit = 10): Promise<{ users: User[]; total: number }> {
    const response = await apiClient.get<User[]>('/users', { page, limit });
    const users = response.data?.map(user => UserSchema.parse(user)) || [];
    return {
      users,
      total: response.pagination?.total || 0,
    };
  },
};
```

### Shared Hooks
```typescript
// packages/shared/src/hooks/useApi.ts
import { useState, useEffect, useCallback } from 'react';
import { ApiResponse } from '../services/api';

type ApiState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
};

export function useApi<T>(
  apiCall: () => Promise<ApiResponse<T>>,
  dependencies: any[] = []
) {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: true,
    error: null,
  });

  const execute = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await apiCall();
      setState({
        data: response.data || null,
        loading: false,
        error: null,
      });
    } catch (error) {
      setState({
        data: null,
        loading: false,
        error: error.message,
      });
    }
  }, dependencies);

  useEffect(() => {
    execute();
  }, [execute]);

  const refetch = useCallback(() => {
    execute();
  }, [execute]);

  return {
    ...state,
    refetch,
  };
}

// packages/shared/src/hooks/useForm.ts
import { useState, useCallback } from 'react';
import { z } from 'zod';

type FormState<T> = {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isValid: boolean;
  isSubmitting: boolean;
};

export function useForm<T extends Record<string, any>>(
  initialValues: T,
  schema?: z.ZodSchema<T>
) {
  const [state, setState] = useState<FormState<T>>({
    values: initialValues,
    errors: {},
    touched: {},
    isValid: true,
    isSubmitting: false,
  });

  const validateField = useCallback((name: keyof T, value: any) => {
    if (!schema) return null;

    try {
      schema.parse({ ...state.values, [name]: value });
      return null;
    } catch (error) {
      if (error instanceof z.ZodError) {
        const fieldError = error.errors.find(err => err.path[0] === name);
        return fieldError?.message || 'Invalid value';
      }
      return 'Validation error';
    }
  }, [schema, state.values]);

  const setValue = useCallback((name: keyof T, value: any) => {
    setState(prev => {
      const newValues = { ...prev.values, [name]: value };
      const error = validateField(name, value);
      const newErrors = { ...prev.errors };
      
      if (error) {
        newErrors[name] = error;
      } else {
        delete newErrors[name];
      }

      return {
        ...prev,
        values: newValues,
        errors: newErrors,
        touched: { ...prev.touched, [name]: true },
        isValid: Object.keys(newErrors).length === 0,
      };
    });
  }, [validateField]);

  const setValues = useCallback((newValues: Partial<T>) => {
    setState(prev => ({
      ...prev,
      values: { ...prev.values, ...newValues },
    }));
  }, []);

  const validateAll = useCallback(() => {
    if (!schema) return true;

    try {
      schema.parse(state.values);
      setState(prev => ({ ...prev, errors: {}, isValid: true }));
      return true;
    } catch (error) {
      if (error instanceof z.ZodError) {
        const errors: Partial<Record<keyof T, string>> = {};
        error.errors.forEach(err => {
          if (err.path[0]) {
            errors[err.path[0] as keyof T] = err.message;
          }
        });
        setState(prev => ({ ...prev, errors, isValid: false }));
      }
      return false;
    }
  }, [schema, state.values]);

  const handleSubmit = useCallback(async (onSubmit: (values: T) => Promise<void>) => {
    setState(prev => ({ ...prev, isSubmitting: true }));

    if (!validateAll()) {
      setState(prev => ({ ...prev, isSubmitting: false }));
      return;
    }

    try {
      await onSubmit(state.values);
    } finally {
      setState(prev => ({ ...prev, isSubmitting: false }));
    }
  }, [state.values, validateAll]);

  const reset = useCallback(() => {
    setState({
      values: initialValues,
      errors: {},
      touched: {},
      isValid: true,
      isSubmitting: false,
    });
  }, [initialValues]);

  return {
    values: state.values,
    errors: state.errors,
    touched: state.touched,
    isValid: state.isValid,
    isSubmitting: state.isSubmitting,
    setValue,
    setValues,
    validateAll,
    handleSubmit,
    reset,
  };
}
```

### Shared Utilities
```typescript
// packages/shared/src/utils/validation.ts
import { z } from 'zod';

export const emailSchema = z.string().email('Invalid email address');
export const passwordSchema = z.string().min(8, 'Password must be at least 8 characters');
export const phoneSchema = z.string().regex(/^\+?[\d\s-()]+$/, 'Invalid phone number');

export const registerSchema = z.object({
  email: emailSchema,
  password: passwordSchema,
  confirmPassword: z.string(),
  name: z.string().min(2, 'Name must be at least 2 characters'),
  phone: phoneSchema.optional(),
}).refine(data => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export const loginSchema = z.object({
  email: emailSchema,
  password: z.string().min(1, 'Password is required'),
});

// packages/shared/src/utils/formatting.ts
import { format, parseISO, isValid } from 'date-fns';

export const formatters = {
  currency: (amount: number, currency = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
    }).format(amount);
  },

  date: (date: string | Date, formatString = 'MMM dd, yyyy') => {
    const dateObj = typeof date === 'string' ? parseISO(date) : date;
    return isValid(dateObj) ? format(dateObj, formatString) : 'Invalid date';
  },

  phone: (phone: string) => {
    const cleaned = phone.replace(/\D/g, '');
    if (cleaned.length === 10) {
      return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6)}`;
    }
    return phone;
  },

  truncate: (text: string, maxLength: number) => {
    if (text.length <= maxLength) return text;
    return `${text.slice(0, maxLength - 3)}...`;
  },
};

// packages/shared/src/utils/storage.ts
export interface StorageAdapter {
  getItem(key: string): Promise<string | null>;
  setItem(key: string, value: string): Promise<void>;
  removeItem(key: string): Promise<void>;
  clear(): Promise<void>;
}

export class Storage {
  constructor(private adapter: StorageAdapter) {}

  async get<T>(key: string, defaultValue?: T): Promise<T | null> {
    try {
      const value = await this.adapter.getItem(key);
      return value ? JSON.parse(value) : defaultValue || null;
    } catch {
      return defaultValue || null;
    }
  }

  async set<T>(key: string, value: T): Promise<void> {
    try {
      await this.adapter.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Storage set error:', error);
    }
  }

  async remove(key: string): Promise<void> {
    try {
      await this.adapter.removeItem(key);
    } catch (error) {
      console.error('Storage remove error:', error);
    }
  }

  async clear(): Promise<void> {
    try {
      await this.adapter.clear();
    } catch (error) {
      console.error('Storage clear error:', error);
    }
  }
}
```

## Common State Management

### Zustand Store
```typescript
// packages/shared/src/stores/authStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User } from '../services/api';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
  updateUser: (updates: Partial<User>) => void;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      login: (user, token) => {
        set({
          user,
          token,
          isAuthenticated: true,
          isLoading: false,
        });
      },

      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },

      updateUser: (updates) => {
        const { user } = get();
        if (user) {
          set({ user: { ...user, ...updates } });
        }
      },

      setLoading: (isLoading) => {
        set({ isLoading });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// packages/shared/src/stores/appStore.ts
interface AppState {
  theme: 'light' | 'dark';
  language: string;
  notifications: boolean;
  setTheme: (theme: 'light' | 'dark') => void;
  setLanguage: (language: string) => void;
  setNotifications: (enabled: boolean) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      theme: 'light',
      language: 'en',
      notifications: true,

      setTheme: (theme) => set({ theme }),
      setLanguage: (language) => set({ language }),
      setNotifications: (notifications) => set({ notifications }),
    }),
    {
      name: 'app-settings',
    }
  )
);
```

### React Query Configuration
```typescript
// packages/shared/src/query/queryClient.ts
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
      retryDelay: 1000,
    },
  },
});

// packages/shared/src/query/userQueries.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { userService, User } from '../services/api';

export const userKeys = {
  all: ['users'] as const,
  lists: () => [...userKeys.all, 'list'] as const,
  list: (filters: Record<string, any>) => [...userKeys.lists(), filters] as const,
  details: () => [...userKeys.all, 'detail'] as const,
  detail: (id: string) => [...userKeys.details(), id] as const,
  current: () => [...userKeys.all, 'current'] as const,
};

export function useCurrentUser() {
  return useQuery({
    queryKey: userKeys.current(),
    queryFn: userService.getCurrentUser,
  });
}

export function useUsers(page = 1, limit = 10) {
  return useQuery({
    queryKey: userKeys.list({ page, limit }),
    queryFn: () => userService.getUsers(page, limit),
  });
}

export function useUpdateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: Partial<User> }) =>
      userService.updateUser(id, updates),
    onSuccess: (updatedUser) => {
      // Update current user cache
      queryClient.setQueryData(userKeys.current(), updatedUser);
      
      // Invalidate user lists
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
    },
  });
}
```

## Universal Components

### Platform-Agnostic Components
```typescript
// packages/shared/src/components/Button.tsx
import React from 'react';

export interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  testID?: string;
}

// This is the shared interface - implementations differ per platform

// packages/mobile/src/components/Button.tsx
import React from 'react';
import { TouchableOpacity, Text, ActivityIndicator, StyleSheet } from 'react-native';
import { ButtonProps } from '@my-app/shared/components/Button';

export const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  testID,
}) => {
  return (
    <TouchableOpacity
      style={[
        styles.button,
        styles[variant],
        styles[size],
        disabled && styles.disabled,
      ]}
      onPress={onPress}
      disabled={disabled || loading}
      testID={testID}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? 'white' : '#007AFF'} />
      ) : (
        <Text style={[styles.text, styles[`${variant}Text`], styles[`${size}Text`]]}>
          {title}
        </Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  primary: {
    backgroundColor: '#007AFF',
  },
  secondary: {
    backgroundColor: '#F2F2F7',
  },
  outline: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  small: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    minHeight: 32,
  },
  medium: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    minHeight: 44,
  },
  large: {
    paddingHorizontal: 24,
    paddingVertical: 14,
    minHeight: 56,
  },
  disabled: {
    opacity: 0.5,
  },
  text: {
    fontWeight: '600',
  },
  primaryText: {
    color: 'white',
  },
  secondaryText: {
    color: '#007AFF',
  },
  outlineText: {
    color: '#007AFF',
  },
  smallText: {
    fontSize: 14,
  },
  mediumText: {
    fontSize: 16,
  },
  largeText: {
    fontSize: 18,
  },
});

// packages/web/src/components/Button.tsx
import React from 'react';
import { ButtonProps } from '@my-app/shared/components/Button';
import './Button.css';

export const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  testID,
}) => {
  return (
    <button
      className={`button button--${variant} button--${size} ${disabled ? 'button--disabled' : ''}`}
      onClick={onPress}
      disabled={disabled || loading}
      data-testid={testID}
    >
      {loading ? (
        <span className="button__spinner" />
      ) : (
        title
      )}
    </button>
  );
};
```

### Theme System
```typescript
// packages/shared/src/theme/index.ts
export interface Theme {
  colors: {
    primary: string;
    secondary: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    error: string;
    warning: string;
    success: string;
  };
  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
  };
  typography: {
    fontSize: {
      xs: number;
      sm: number;
      md: number;
      lg: number;
      xl: number;
    };
    fontWeight: {
      normal: string;
      medium: string;
      bold: string;
    };
  };
  borderRadius: {
    sm: number;
    md: number;
    lg: number;
  };
}

export const lightTheme: Theme = {
  colors: {
    primary: '#007AFF',
    secondary: '#5856D6',
    background: '#FFFFFF',
    surface: '#F2F2F7',
    text: '#000000',
    textSecondary: '#6D6D80',
    border: '#C6C6C8',
    error: '#FF3B30',
    warning: '#FF9500',
    success: '#34C759',
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
  },
  typography: {
    fontSize: {
      xs: 12,
      sm: 14,
      md: 16,
      lg: 18,
      xl: 24,
    },
    fontWeight: {
      normal: '400',
      medium: '500',
      bold: '700',
    },
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 12,
  },
};

export const darkTheme: Theme = {
  ...lightTheme,
  colors: {
    ...lightTheme.colors,
    background: '#000000',
    surface: '#1C1C1E',
    text: '#FFFFFF',
    textSecondary: '#8E8E93',
    border: '#38383A',
  },
};

// Theme provider
import React, { createContext, useContext } from 'react';
import { useAppStore } from '../stores/appStore';

const ThemeContext = createContext<Theme>(lightTheme);

export const useTheme = () => useContext(ThemeContext);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { theme: themeName } = useAppStore();
  const theme = themeName === 'dark' ? darkTheme : lightTheme;

  return (
    <ThemeContext.Provider value={theme}>
      {children}
    </ThemeContext.Provider>
  );
};
```

## Platform-Specific Implementations

### Storage Adapters
```typescript
// packages/mobile/src/platform/storage.ts
import AsyncStorage from '@react-native-async-storage/async-storage';
import { StorageAdapter } from '@my-app/shared/utils/storage';

export const mobileStorageAdapter: StorageAdapter = {
  async getItem(key: string): Promise<string | null> {
    return AsyncStorage.getItem(key);
  },

  async setItem(key: string, value: string): Promise<void> {
    await AsyncStorage.setItem(key, value);
  },

  async removeItem(key: string): Promise<void> {
    await AsyncStorage.removeItem(key);
  },

  async clear(): Promise<void> {
    await AsyncStorage.clear();
  },
};

// packages/web/src/platform/storage.ts
import { StorageAdapter } from '@my-app/shared/utils/storage';

export const webStorageAdapter: StorageAdapter = {
  async getItem(key: string): Promise<string | null> {
    return localStorage.getItem(key);
  },

  async setItem(key: string, value: string): Promise<void> {
    localStorage.setItem(key, value);
  },

  async removeItem(key: string): Promise<void> {
    localStorage.removeItem(key);
  },

  async clear(): Promise<void> {
    localStorage.clear();
  },
};
```

### Navigation Abstractions
```typescript
// packages/shared/src/navigation/types.ts
export interface NavigationService {
  navigate(route: string, params?: any): void;
  goBack(): void;
  reset(route: string, params?: any): void;
  canGoBack(): boolean;
}

// packages/mobile/src/platform/navigation.ts
import { NavigationContainer, NavigationContainerRef } from '@react-navigation/native';
import { NavigationService } from '@my-app/shared/navigation/types';

class MobileNavigationService implements NavigationService {
  private navigationRef: React.RefObject<NavigationContainerRef<any>>;

  constructor() {
    this.navigationRef = React.createRef();
  }

  navigate(route: string, params?: any): void {
    this.navigationRef.current?.navigate(route, params);
  }

  goBack(): void {
    this.navigationRef.current?.goBack();
  }

  reset(route: string, params?: any): void {
    this.navigationRef.current?.reset({
      index: 0,
      routes: [{ name: route, params }],
    });
  }

  canGoBack(): boolean {
    return this.navigationRef.current?.canGoBack() || false;
  }

  getNavigationRef() {
    return this.navigationRef;
  }
}

export const navigationService = new MobileNavigationService();

// packages/web/src/platform/navigation.ts
import { useNavigate, useLocation } from 'react-router-dom';
import { NavigationService } from '@my-app/shared/navigation/types';

class WebNavigationService implements NavigationService {
  private navigate: ((path: string, options?: any) => void) | null = null;
  private location: any = null;

  setNavigate(navigate: any, location: any) {
    this.navigate = navigate;
    this.location = location;
  }

  navigate(route: string, params?: any): void {
    if (this.navigate) {
      const search = params ? `?${new URLSearchParams(params)}` : '';
      this.navigate(`${route}${search}`);
    }
  }

  goBack(): void {
    if (this.navigate) {
      this.navigate(-1);
    }
  }

  reset(route: string, params?: any): void {
    this.navigate(route, { replace: true, state: params });
  }

  canGoBack(): boolean {
    return window.history.length > 1;
  }
}

export const navigationService = new WebNavigationService();

// Hook to initialize navigation service
export const useNavigationService = () => {
  const navigate = useNavigate();
  const location = useLocation();

  React.useEffect(() => {
    navigationService.setNavigate(navigate, location);
  }, [navigate, location]);

  return navigationService;
};
```

## Testing Strategies

### Shared Test Utilities
```typescript
// packages/shared/src/testing/testUtils.ts
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '../theme';

// Create a custom render function that includes providers
export const createTestUtils = (platformSpecificProviders: React.FC<any> = React.Fragment) => {
  const AllTheProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    return (
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <platformSpecificProviders>
            {children}
          </platformSpecificProviders>
        </ThemeProvider>
      </QueryClientProvider>
    );
  };

  const customRender = (ui: React.ReactElement, options?: Omit<RenderOptions, 'wrapper'>) =>
    render(ui, { wrapper: AllTheProviders, ...options });

  return {
    render: customRender,
    AllTheProviders,
  };
};

// Mock factories
export const createMockUser = (overrides = {}) => ({
  id: '1',
  email: 'test@example.com',
  name: 'Test User',
  createdAt: new Date().toISOString(),
  ...overrides,
});

export const createMockApiResponse = <T>(data: T, overrides = {}) => ({
  success: true,
  data,
  ...overrides,
});

// Test data factories
export const testData = {
  user: createMockUser,
  apiResponse: createMockApiResponse,
};
```

### Platform-Specific Test Setup
```javascript
// packages/mobile/src/testing/setup.js
import 'react-native-gesture-handler/jestSetup';
import mockAsyncStorage from '@react-native-async-storage/async-storage/jest/async-storage-mock';

jest.mock('@react-native-async-storage/async-storage', () => mockAsyncStorage);

jest.mock('react-native-reanimated', () => {
  const Reanimated = require('react-native-reanimated/mock');
  Reanimated.default.call = () => {};
  return Reanimated;
});

// Mock native modules
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  return {
    ...RN,
    NativeModules: {
      ...RN.NativeModules,
      CustomModule: {
        calculateSum: jest.fn(),
        processData: jest.fn(),
      },
    },
  };
});

// packages/web/src/testing/setup.js
import '@testing-library/jest-dom';

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// Mock fetch
global.fetch = jest.fn();

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));
```

---

*Continue to: [19-best-practices.md](./19-best-practices.md)*
