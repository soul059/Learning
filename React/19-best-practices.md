# Best Practices and Guidelines

## Table of Contents
- [Code Quality Standards](#code-quality-standards)
- [Performance Best Practices](#performance-best-practices)
- [Security Guidelines](#security-guidelines)
- [Architecture Patterns](#architecture-patterns)
- [Development Workflow](#development-workflow)
- [Code Review Guidelines](#code-review-guidelines)
- [Documentation Standards](#documentation-standards)
- [Deployment Best Practices](#deployment-best-practices)

## Code Quality Standards

### TypeScript Best Practices
```typescript
// ✅ Good: Strict typing with proper interfaces
interface User {
  readonly id: string;
  email: string;
  name: string;
  avatar?: string;
  createdAt: Date;
  updatedAt: Date;
}

interface UserRepository {
  findById(id: string): Promise<User | null>;
  create(userData: Omit<User, 'id' | 'createdAt' | 'updatedAt'>): Promise<User>;
  update(id: string, updates: Partial<Pick<User, 'email' | 'name' | 'avatar'>>): Promise<User>;
  delete(id: string): Promise<void>;
}

// ✅ Good: Generic utility types
type ApiResponse<TData> = {
  success: true;
  data: TData;
  pagination?: PaginationInfo;
} | {
  success: false;
  error: string;
  code: number;
};

type AsyncState<TData, TError = string> = {
  data: TData | null;
  loading: boolean;
  error: TError | null;
};

// ✅ Good: Branded types for type safety
type UserId = string & { readonly __brand: 'UserId' };
type Email = string & { readonly __brand: 'Email' };

const createUserId = (id: string): UserId => id as UserId;
const createEmail = (email: string): Email => {
  if (!email.includes('@')) {
    throw new Error('Invalid email format');
  }
  return email as Email;
};

// ❌ Bad: Using `any` or loose typing
const processUserData = (data: any) => {
  return data.user.profile.settings; // No type safety
};

// ✅ Good: Proper error handling with discriminated unions
type Result<TSuccess, TError> = 
  | { success: true; data: TSuccess }
  | { success: false; error: TError };

const fetchUser = async (id: UserId): Promise<Result<User, string>> => {
  try {
    const user = await userRepository.findById(id);
    if (!user) {
      return { success: false, error: 'User not found' };
    }
    return { success: true, data: user };
  } catch (error) {
    return { success: false, error: error.message };
  }
};
```

### Component Architecture
```typescript
// ✅ Good: Compound component pattern
interface TabsContextType {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const TabsContext = React.createContext<TabsContextType | null>(null);

const useTabs = () => {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error('useTabs must be used within Tabs component');
  }
  return context;
};

interface TabsProps {
  defaultTab?: string;
  onTabChange?: (tab: string) => void;
  children: React.ReactNode;
}

const Tabs: React.FC<TabsProps> & {
  List: typeof TabsList;
  Tab: typeof Tab;
  Panels: typeof TabsPanels;
  Panel: typeof TabPanel;
} = ({ defaultTab = '', onTabChange, children }) => {
  const [activeTab, setActiveTab] = useState(defaultTab);

  const handleTabChange = useCallback((tab: string) => {
    setActiveTab(tab);
    onTabChange?.(tab);
  }, [onTabChange]);

  const value = useMemo(() => ({
    activeTab,
    setActiveTab: handleTabChange,
  }), [activeTab, handleTabChange]);

  return (
    <TabsContext.Provider value={value}>
      <div className="tabs">
        {children}
      </div>
    </TabsContext.Provider>
  );
};

const TabsList: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="tabs-list" role="tablist">
    {children}
  </div>
);

const Tab: React.FC<{ id: string; children: React.ReactNode }> = ({ id, children }) => {
  const { activeTab, setActiveTab } = useTabs();
  const isActive = activeTab === id;

  return (
    <button
      className={`tab ${isActive ? 'tab--active' : ''}`}
      role="tab"
      aria-selected={isActive}
      onClick={() => setActiveTab(id)}
    >
      {children}
    </button>
  );
};

const TabsPanels: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="tabs-panels">
    {children}
  </div>
);

const TabPanel: React.FC<{ id: string; children: React.ReactNode }> = ({ id, children }) => {
  const { activeTab } = useTabs();
  const isActive = activeTab === id;

  if (!isActive) return null;

  return (
    <div className="tab-panel" role="tabpanel">
      {children}
    </div>
  );
};

Tabs.List = TabsList;
Tabs.Tab = Tab;
Tabs.Panels = TabsPanels;
Tabs.Panel = TabPanel;

// Usage
const MyTabs = () => (
  <Tabs defaultTab="profile" onTabChange={console.log}>
    <Tabs.List>
      <Tabs.Tab id="profile">Profile</Tabs.Tab>
      <Tabs.Tab id="settings">Settings</Tabs.Tab>
      <Tabs.Tab id="billing">Billing</Tabs.Tab>
    </Tabs.List>
    <Tabs.Panels>
      <Tabs.Panel id="profile">
        <ProfileContent />
      </Tabs.Panel>
      <Tabs.Panel id="settings">
        <SettingsContent />
      </Tabs.Panel>
      <Tabs.Panel id="billing">
        <BillingContent />
      </Tabs.Panel>
    </Tabs.Panels>
  </Tabs>
);
```

### Custom Hooks Best Practices
```typescript
// ✅ Good: Focused, reusable custom hook
interface UseAsyncOptions<TData> {
  immediate?: boolean;
  onSuccess?: (data: TData) => void;
  onError?: (error: Error) => void;
  initialData?: TData;
}

function useAsync<TData>(
  asyncFn: () => Promise<TData>,
  deps: React.DependencyList,
  options: UseAsyncOptions<TData> = {}
) {
  const {
    immediate = true,
    onSuccess,
    onError,
    initialData = null,
  } = options;

  const [state, setState] = useState<AsyncState<TData>>({
    data: initialData,
    loading: false,
    error: null,
  });

  const execute = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await asyncFn();
      setState({ data, loading: false, error: null });
      onSuccess?.(data);
      return data;
    } catch (error) {
      const errorObj = error instanceof Error ? error : new Error(String(error));
      setState({ data: null, loading: false, error: errorObj.message });
      onError?.(errorObj);
      throw errorObj;
    }
  }, deps);

  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);

  const reset = useCallback(() => {
    setState({ data: initialData, loading: false, error: null });
  }, [initialData]);

  return {
    ...state,
    execute,
    reset,
  };
}

// ✅ Good: Hook with proper cleanup
function useEventListener<K extends keyof WindowEventMap>(
  eventName: K,
  handler: (event: WindowEventMap[K]) => void,
  element: EventTarget = window
) {
  const savedHandler = useRef(handler);

  useEffect(() => {
    savedHandler.current = handler;
  }, [handler]);

  useEffect(() => {
    const eventListener = (event: Event) => savedHandler.current(event as WindowEventMap[K]);
    element.addEventListener(eventName, eventListener);
    
    return () => {
      element.removeEventListener(eventName, eventListener);
    };
  }, [eventName, element]);
}

// ✅ Good: Hook composition
function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error('Error reading localStorage key:', key, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error('Error setting localStorage key:', key, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue] as const;
}

function usePersistedState<T>(key: string, initialValue: T) {
  const [value, setValue] = useLocalStorage(key, initialValue);
  
  // Sync across tabs
  useEventListener('storage', (e) => {
    if (e.key === key && e.newValue) {
      try {
        setValue(JSON.parse(e.newValue));
      } catch (error) {
        console.error('Error parsing storage event:', error);
      }
    }
  });

  return [value, setValue] as const;
}
```

## Performance Best Practices

### React Performance Optimization
```typescript
// ✅ Good: Memoization strategies
interface ExpensiveComponentProps {
  data: ComplexData[];
  onItemClick: (id: string) => void;
  filters: FilterOptions;
}

const ExpensiveComponent = React.memo<ExpensiveComponentProps>(({ 
  data, 
  onItemClick, 
  filters 
}) => {
  // Memoize expensive calculations
  const processedData = useMemo(() => {
    return data
      .filter(item => applyFilters(item, filters))
      .sort((a, b) => a.priority - b.priority)
      .slice(0, 100); // Limit rendering
  }, [data, filters]);

  // Memoize handlers to prevent child re-renders
  const handleItemClick = useCallback((id: string) => {
    onItemClick(id);
  }, [onItemClick]);

  return (
    <VirtualizedList
      items={processedData}
      renderItem={({ item }) => (
        <ListItem 
          key={item.id}
          item={item}
          onClick={handleItemClick}
        />
      )}
      itemHeight={60}
      windowSize={10}
    />
  );
}, (prevProps, nextProps) => {
  // Custom comparison for complex props
  return (
    prevProps.data === nextProps.data &&
    prevProps.onItemClick === nextProps.onItemClick &&
    shallowEqual(prevProps.filters, nextProps.filters)
  );
});

// ✅ Good: Code splitting with lazy loading
const LazyComponent = React.lazy(() => 
  import('./HeavyComponent').then(module => ({
    default: module.HeavyComponent
  }))
);

const AppWithSuspense = () => (
  <Suspense fallback={<LoadingSpinner />}>
    <Routes>
      <Route path="/heavy" element={<LazyComponent />} />
    </Routes>
  </Suspense>
);

// ✅ Good: Optimistic updates with error boundaries
interface OptimisticUpdateHookOptions<T> {
  mutationFn: (data: T) => Promise<T>;
  onSuccess?: (data: T) => void;
  onError?: (error: Error, originalData: T) => void;
}

function useOptimisticUpdate<T>(
  currentData: T,
  options: OptimisticUpdateHookOptions<T>
) {
  const [optimisticData, setOptimisticData] = useState(currentData);
  const [isUpdating, setIsUpdating] = useState(false);

  const updateOptimistically = useCallback(async (newData: T) => {
    const originalData = optimisticData;
    setOptimisticData(newData);
    setIsUpdating(true);

    try {
      const result = await options.mutationFn(newData);
      setOptimisticData(result);
      options.onSuccess?.(result);
    } catch (error) {
      setOptimisticData(originalData);
      options.onError?.(error as Error, originalData);
    } finally {
      setIsUpdating(false);
    }
  }, [optimisticData, options]);

  useEffect(() => {
    setOptimisticData(currentData);
  }, [currentData]);

  return {
    data: optimisticData,
    isUpdating,
    updateOptimistically,
  };
}
```

### Bundle Optimization
```typescript
// ✅ Good: Tree-shakable exports
// utils/index.ts - Avoid barrel exports for large utilities
export { formatDate } from './date';
export { validateEmail } from './validation';
export { debounce } from './timing';

// ✅ Good: Conditional loading
const loadAnalytics = async () => {
  if (process.env.NODE_ENV === 'production') {
    const { initAnalytics } = await import('./analytics');
    return initAnalytics();
  }
  return null;
};

// ✅ Good: Resource hints
const PreloadCriticalResources = () => (
  <Head>
    <link 
      rel="preload" 
      href="/fonts/inter-var.woff2" 
      as="font" 
      type="font/woff2" 
      crossOrigin="anonymous" 
    />
    <link 
      rel="prefetch" 
      href="/api/user/profile" 
      as="fetch" 
      crossOrigin="anonymous" 
    />
    <link 
      rel="modulepreload" 
      href="/js/chart-library.js" 
    />
  </Head>
);

// ✅ Good: Image optimization
interface OptimizedImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  priority?: boolean;
  sizes?: string;
}

const OptimizedImage: React.FC<OptimizedImageProps> = ({
  src,
  alt,
  width,
  height,
  priority = false,
  sizes = '100vw',
}) => {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);

  const webpSrc = src.replace(/\.(jpg|jpeg|png)$/i, '.webp');
  const avifSrc = src.replace(/\.(jpg|jpeg|png)$/i, '.avif');

  return (
    <picture>
      <source srcSet={avifSrc} type="image/avif" />
      <source srcSet={webpSrc} type="image/webp" />
      <img
        src={src}
        alt={alt}
        width={width}
        height={height}
        loading={priority ? 'eager' : 'lazy'}
        decoding="async"
        sizes={sizes}
        onLoad={() => setLoaded(true)}
        onError={() => setError(true)}
        style={{
          opacity: loaded ? 1 : 0,
          transition: 'opacity 0.3s ease',
        }}
      />
    </picture>
  );
};
```

## Security Guidelines

### Input Validation and Sanitization
```typescript
// ✅ Good: Comprehensive input validation
import { z } from 'zod';
import DOMPurify from 'dompurify';

// Validation schemas
const userInputSchema = z.object({
  email: z.string().email().max(254),
  name: z.string().min(1).max(100).regex(/^[a-zA-Z\s-']+$/),
  bio: z.string().max(500).optional(),
  website: z.string().url().optional(),
  age: z.number().int().min(13).max(120),
});

// XSS prevention
const sanitizeHtml = (html: string): string => {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'p', 'br', 'ul', 'ol', 'li'],
    ALLOWED_ATTR: [],
  });
};

// SQL injection prevention (for API layer)
const createParameterizedQuery = (query: string, params: any[]) => {
  // Use proper ORM or prepared statements
  return {
    text: query,
    values: params,
  };
};

// ✅ Good: Secure API client
class SecureApiClient {
  private apiKey: string;
  private baseURL: string;
  private csrfToken: string | null = null;

  constructor(apiKey: string, baseURL: string) {
    this.apiKey = apiKey;
    this.baseURL = baseURL;
    this.loadCSRFToken();
  }

  private async loadCSRFToken() {
    try {
      const response = await fetch(`${this.baseURL}/csrf-token`, {
        credentials: 'include',
      });
      const data = await response.json();
      this.csrfToken = data.token;
    } catch (error) {
      console.error('Failed to load CSRF token:', error);
    }
  }

  private getHeaders(): Headers {
    const headers = new Headers({
      'Content-Type': 'application/json',
      'X-API-Key': this.apiKey,
    });

    if (this.csrfToken) {
      headers.set('X-CSRF-Token', this.csrfToken);
    }

    return headers;
  }

  async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: this.getHeaders(),
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    return response.json();
  }
}

// ✅ Good: Content Security Policy
const CSP_HEADER = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline' https://cdn.example.com",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "img-src 'self' data: https:",
  "font-src 'self' https://fonts.gstatic.com",
  "connect-src 'self' https://api.example.com",
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join('; ');
```

### Authentication and Authorization
```typescript
// ✅ Good: Secure token management
interface TokenManager {
  getAccessToken(): string | null;
  getRefreshToken(): string | null;
  setTokens(accessToken: string, refreshToken: string): void;
  clearTokens(): void;
  isTokenExpired(token: string): boolean;
  refreshAccessToken(): Promise<string>;
}

class SecureTokenManager implements TokenManager {
  private readonly ACCESS_TOKEN_KEY = 'access_token';
  private readonly REFRESH_TOKEN_KEY = 'refresh_token';

  getAccessToken(): string | null {
    return this.getSecureItem(this.ACCESS_TOKEN_KEY);
  }

  getRefreshToken(): string | null {
    return this.getSecureItem(this.REFRESH_TOKEN_KEY);
  }

  setTokens(accessToken: string, refreshToken: string): void {
    this.setSecureItem(this.ACCESS_TOKEN_KEY, accessToken);
    this.setSecureItem(this.REFRESH_TOKEN_KEY, refreshToken);
  }

  clearTokens(): void {
    this.removeSecureItem(this.ACCESS_TOKEN_KEY);
    this.removeSecureItem(this.REFRESH_TOKEN_KEY);
  }

  isTokenExpired(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.exp * 1000 < Date.now();
    } catch {
      return true;
    }
  }

  async refreshAccessToken(): Promise<string> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken || this.isTokenExpired(refreshToken)) {
      throw new Error('No valid refresh token');
    }

    const response = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${refreshToken}`,
      },
    });

    if (!response.ok) {
      throw new Error('Token refresh failed');
    }

    const { accessToken, refreshToken: newRefreshToken } = await response.json();
    this.setTokens(accessToken, newRefreshToken);
    return accessToken;
  }

  private getSecureItem(key: string): string | null {
    try {
      // In production, use secure storage
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  private setSecureItem(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch (error) {
      console.error('Failed to store secure item:', error);
    }
  }

  private removeSecureItem(key: string): void {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error('Failed to remove secure item:', error);
    }
  }
}

// ✅ Good: Permission-based access control
interface Permission {
  resource: string;
  action: 'create' | 'read' | 'update' | 'delete';
}

interface Role {
  id: string;
  name: string;
  permissions: Permission[];
}

interface User {
  id: string;
  email: string;
  roles: Role[];
}

class PermissionManager {
  private user: User | null = null;

  setUser(user: User) {
    this.user = user;
  }

  hasPermission(resource: string, action: Permission['action']): boolean {
    if (!this.user) return false;

    return this.user.roles.some(role =>
      role.permissions.some(permission =>
        permission.resource === resource && permission.action === action
      )
    );
  }

  hasRole(roleName: string): boolean {
    if (!this.user) return false;
    return this.user.roles.some(role => role.name === roleName);
  }

  canAccess(requiredPermissions: Permission[]): boolean {
    return requiredPermissions.every(permission =>
      this.hasPermission(permission.resource, permission.action)
    );
  }
}

// ✅ Good: Protected route component
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPermissions?: Permission[];
  fallback?: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredPermissions = [],
  fallback = <div>Access Denied</div>,
}) => {
  const { user } = useAuth();
  const permissionManager = new PermissionManager();
  
  if (user) {
    permissionManager.setUser(user);
  }

  if (!user) {
    return <Navigate to="/login" />;
  }

  if (requiredPermissions.length > 0 && !permissionManager.canAccess(requiredPermissions)) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};
```

## Architecture Patterns

### Clean Architecture Implementation
```typescript
// Domain layer - Business entities
interface User {
  readonly id: UserId;
  readonly email: Email;
  readonly name: string;
  readonly createdAt: Date;
}

interface UserRepository {
  findById(id: UserId): Promise<User | null>;
  findByEmail(email: Email): Promise<User | null>;
  save(user: User): Promise<void>;
  delete(id: UserId): Promise<void>;
}

// Use cases layer - Business logic
class CreateUserUseCase {
  constructor(
    private userRepository: UserRepository,
    private emailService: EmailService,
    private logger: Logger
  ) {}

  async execute(input: CreateUserInput): Promise<Result<User, CreateUserError>> {
    try {
      // Validate input
      const validationResult = await this.validateInput(input);
      if (!validationResult.isValid) {
        return Result.failure(new CreateUserError(validationResult.errors));
      }

      // Check if user already exists
      const existingUser = await this.userRepository.findByEmail(input.email);
      if (existingUser) {
        return Result.failure(new CreateUserError('User already exists'));
      }

      // Create user
      const user = new User({
        id: generateUserId(),
        email: input.email,
        name: input.name,
        createdAt: new Date(),
      });

      // Save user
      await this.userRepository.save(user);

      // Send welcome email
      await this.emailService.sendWelcomeEmail(user);

      this.logger.info('User created successfully', { userId: user.id });

      return Result.success(user);
    } catch (error) {
      this.logger.error('Failed to create user', { error, input });
      return Result.failure(new CreateUserError('Internal server error'));
    }
  }

  private async validateInput(input: CreateUserInput): Promise<ValidationResult> {
    // Validation logic
    return { isValid: true, errors: [] };
  }
}

// Infrastructure layer - External concerns
class ApiUserRepository implements UserRepository {
  constructor(private httpClient: HttpClient) {}

  async findById(id: UserId): Promise<User | null> {
    try {
      const response = await this.httpClient.get(`/users/${id}`);
      return response.data ? this.mapToUser(response.data) : null;
    } catch (error) {
      if (error.status === 404) return null;
      throw error;
    }
  }

  async save(user: User): Promise<void> {
    await this.httpClient.post('/users', this.mapToDto(user));
  }

  private mapToUser(dto: any): User {
    return new User({
      id: dto.id,
      email: dto.email,
      name: dto.name,
      createdAt: new Date(dto.createdAt),
    });
  }

  private mapToDto(user: User): any {
    return {
      id: user.id,
      email: user.email,
      name: user.name,
      createdAt: user.createdAt.toISOString(),
    };
  }
}

// Presentation layer - React components
const CreateUserForm: React.FC = () => {
  const [createUser] = useCreateUserMutation();
  const { register, handleSubmit, formState: { errors } } = useForm<CreateUserInput>();

  const onSubmit = async (data: CreateUserInput) => {
    const result = await createUser(data);
    
    if (result.isSuccess) {
      toast.success('User created successfully');
    } else {
      toast.error(result.error.message);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        {...register('email', { required: 'Email is required' })}
        type="email"
        placeholder="Email"
      />
      {errors.email && <span>{errors.email.message}</span>}
      
      <input
        {...register('name', { required: 'Name is required' })}
        placeholder="Name"
      />
      {errors.name && <span>{errors.name.message}</span>}
      
      <button type="submit">Create User</button>
    </form>
  );
};

// Dependency injection container
class Container {
  private services = new Map<string, any>();

  register<T>(key: string, factory: () => T): void {
    this.services.set(key, factory);
  }

  resolve<T>(key: string): T {
    const factory = this.services.get(key);
    if (!factory) {
      throw new Error(`Service ${key} not found`);
    }
    return factory();
  }
}

// Service registration
const container = new Container();

container.register('httpClient', () => new HttpClient());
container.register('userRepository', () => 
  new ApiUserRepository(container.resolve('httpClient'))
);
container.register('emailService', () => new EmailService());
container.register('logger', () => new Logger());
container.register('createUserUseCase', () => 
  new CreateUserUseCase(
    container.resolve('userRepository'),
    container.resolve('emailService'),
    container.resolve('logger')
  )
);
```

## Development Workflow

### Git Workflow and Conventions
```bash
# ✅ Good: Conventional commit messages
git commit -m "feat(auth): add user registration with email verification"
git commit -m "fix(api): handle network timeout errors properly"
git commit -m "docs(readme): update installation instructions"
git commit -m "refactor(components): extract reusable Button component"
git commit -m "test(user-service): add unit tests for user creation"
git commit -m "chore(deps): update React to v18.2.0"

# ✅ Good: Feature branch workflow
git checkout -b feature/user-authentication
git checkout -b bugfix/login-form-validation
git checkout -b hotfix/security-vulnerability

# ✅ Good: Interactive rebase for clean history
git rebase -i HEAD~3  # Squash related commits
git rebase main       # Keep feature branch up to date

# ✅ Good: Git hooks for code quality
# .git/hooks/pre-commit
#!/bin/sh
npm run lint
npm run type-check
npm run test:changed

# .git/hooks/commit-msg
#!/bin/sh
npx commitlint --edit $1
```

### Continuous Integration Configuration
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '18'
  CACHE_VERSION: v1

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run linting
        run: npm run lint
      
      - name: Run type checking
        run: npm run type-check
      
      - name: Run unit tests
        run: npm run test:coverage
      
      - name: Run E2E tests
        run: npm run test:e2e
        env:
          CI: true
      
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/lcov.info
          flags: unittests
          name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run security audit
        run: npm audit --production --audit-level=moderate
      
      - name: Run SAST scan
        uses: github/super-linter@v4
        env:
          DEFAULT_BRANCH: main
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build application
        run: npm run build
        env:
          NODE_ENV: production
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: build-files
          path: dist/

  deploy:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    
    steps:
      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: build-files
          path: dist/
      
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Add deployment commands here
```

---

*Continue to: [20-troubleshooting.md](./20-troubleshooting.md)*
