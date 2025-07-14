# Effective Java for Android Development

## Table of Contents
- [Overview](#overview)
- [Object Creation and Destruction](#object-creation-and-destruction)
- [Methods Common to All Objects](#methods-common-to-all-objects)
- [Classes and Interfaces](#classes-and-interfaces)
- [Generics and Collections](#generics-and-collections)
- [Enums and Annotations](#enums-and-annotations)
- [Lambdas and Streams](#lambdas-and-streams)
- [Exceptions and Error Handling](#exceptions-and-error-handling)
- [Concurrency in Android](#concurrency-in-android)
- [Performance Optimization](#performance-optimization)
- [Android-Specific Best Practices](#android-specific-best-practices)

## Overview

Effective Java principles applied to Android development help create maintainable, efficient, and robust applications. This guide covers essential Java patterns and practices specifically relevant to Android development.

### Key Principles for Android
- **Minimize object creation**: Reduce GC pressure
- **Prefer composition over inheritance**: Flexible designs
- **Use immutable objects**: Thread safety and reliability
- **Handle resources properly**: Prevent memory leaks
- **Design for testability**: Clean, testable code

## Object Creation and Destruction

### Builder Pattern for Complex Objects
```java
public class NetworkRequest {
    private final String url;
    private final String method;
    private final Map<String, String> headers;
    private final String body;
    private final int timeout;
    private final boolean cache;

    private NetworkRequest(Builder builder) {
        this.url = builder.url;
        this.method = builder.method;
        this.headers = Collections.unmodifiableMap(new HashMap<>(builder.headers));
        this.body = builder.body;
        this.timeout = builder.timeout;
        this.cache = builder.cache;
    }

    public static class Builder {
        // Required parameters
        private final String url;
        
        // Optional parameters with defaults
        private String method = "GET";
        private Map<String, String> headers = new HashMap<>();
        private String body = "";
        private int timeout = 30000;
        private boolean cache = true;

        public Builder(String url) {
            this.url = Objects.requireNonNull(url, "URL cannot be null");
        }

        public Builder method(String method) {
            this.method = Objects.requireNonNull(method, "Method cannot be null");
            return this;
        }

        public Builder addHeader(String key, String value) {
            this.headers.put(
                Objects.requireNonNull(key, "Header key cannot be null"),
                Objects.requireNonNull(value, "Header value cannot be null")
            );
            return this;
        }

        public Builder headers(Map<String, String> headers) {
            this.headers.clear();
            if (headers != null) {
                this.headers.putAll(headers);
            }
            return this;
        }

        public Builder body(String body) {
            this.body = body != null ? body : "";
            return this;
        }

        public Builder timeout(int timeout) {
            if (timeout <= 0) {
                throw new IllegalArgumentException("Timeout must be positive");
            }
            this.timeout = timeout;
            return this;
        }

        public Builder cache(boolean cache) {
            this.cache = cache;
            return this;
        }

        public NetworkRequest build() {
            // Validation
            if (url.trim().isEmpty()) {
                throw new IllegalStateException("URL cannot be empty");
            }
            
            return new NetworkRequest(this);
        }
    }

    // Getters
    public String getUrl() { return url; }
    public String getMethod() { return method; }
    public Map<String, String> getHeaders() { return headers; }
    public String getBody() { return body; }
    public int getTimeout() { return timeout; }
    public boolean isCacheEnabled() { return cache; }

    @Override
    public String toString() {
        return "NetworkRequest{" +
                "url='" + url + '\'' +
                ", method='" + method + '\'' +
                ", headers=" + headers +
                ", timeout=" + timeout +
                ", cache=" + cache +
                '}';
    }
}

// Usage in Android
public class ApiService {
    
    public void fetchUserProfile(String userId) {
        NetworkRequest request = new NetworkRequest.Builder("https://api.example.com/users/" + userId)
                .method("GET")
                .addHeader("Authorization", "Bearer " + getAuthToken())
                .addHeader("Accept", "application/json")
                .timeout(15000)
                .cache(true)
                .build();
                
        executeRequest(request);
    }
    
    public void updateUserProfile(String userId, UserProfile profile) {
        NetworkRequest request = new NetworkRequest.Builder("https://api.example.com/users/" + userId)
                .method("PUT")
                .addHeader("Authorization", "Bearer " + getAuthToken())
                .addHeader("Content-Type", "application/json")
                .body(profileToJson(profile))
                .timeout(30000)
                .cache(false)
                .build();
                
        executeRequest(request);
    }
}
```

### Factory Pattern for Object Creation
```java
public abstract class DatabaseHelper {
    
    public static DatabaseHelper getInstance(Context context, DatabaseType type) {
        switch (type) {
            case SQLITE:
                return new SQLiteDatabaseHelper(context);
            case ROOM:
                return new RoomDatabaseHelper(context);
            case REALM:
                return new RealmDatabaseHelper(context);
            default:
                throw new IllegalArgumentException("Unsupported database type: " + type);
        }
    }
    
    public abstract void insert(String table, ContentValues values);
    public abstract Cursor query(String table, String[] columns, String selection);
    public abstract void update(String table, ContentValues values, String whereClause);
    public abstract void delete(String table, String whereClause);
}

public enum DatabaseType {
    SQLITE, ROOM, REALM
}

// Concrete implementations
public class SQLiteDatabaseHelper extends DatabaseHelper {
    private SQLiteOpenHelper helper;
    
    public SQLiteDatabaseHelper(Context context) {
        this.helper = new MySQLiteOpenHelper(context);
    }
    
    @Override
    public void insert(String table, ContentValues values) {
        SQLiteDatabase db = helper.getWritableDatabase();
        try {
            db.insert(table, null, values);
        } finally {
            db.close();
        }
    }
    
    // Other method implementations...
}
```

### Singleton Pattern for Android Services
```java
public class PreferencesManager {
    private static volatile PreferencesManager instance;
    private final SharedPreferences preferences;
    private final SharedPreferences.Editor editor;

    private PreferencesManager(Context context) {
        preferences = context.getApplicationContext()
                .getSharedPreferences("app_prefs", Context.MODE_PRIVATE);
        editor = preferences.edit();
    }

    public static PreferencesManager getInstance(Context context) {
        if (instance == null) {
            synchronized (PreferencesManager.class) {
                if (instance == null) {
                    instance = new PreferencesManager(context);
                }
            }
        }
        return instance;
    }

    public void putString(String key, String value) {
        editor.putString(key, value);
        editor.apply();
    }

    public String getString(String key, String defaultValue) {
        return preferences.getString(key, defaultValue);
    }

    public void putBoolean(String key, boolean value) {
        editor.putBoolean(key, value);
        editor.apply();
    }

    public boolean getBoolean(String key, boolean defaultValue) {
        return preferences.getBoolean(key, defaultValue);
    }

    public void putInt(String key, int value) {
        editor.putInt(key, value);
        editor.apply();
    }

    public int getInt(String key, int defaultValue) {
        return preferences.getInt(key, defaultValue);
    }

    public void remove(String key) {
        editor.remove(key);
        editor.apply();
    }

    public void clear() {
        editor.clear();
        editor.apply();
    }
}
```

## Methods Common to All Objects

### Proper equals() and hashCode() Implementation
```java
public class User {
    private final long id;
    private final String email;
    private final String name;
    private final Date createdAt;

    public User(long id, String email, String name, Date createdAt) {
        this.id = id;
        this.email = Objects.requireNonNull(email, "Email cannot be null");
        this.name = Objects.requireNonNull(name, "Name cannot be null");
        this.createdAt = new Date(Objects.requireNonNull(createdAt, "Created date cannot be null").getTime());
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        User user = (User) obj;
        return id == user.id &&
               Objects.equals(email, user.email) &&
               Objects.equals(name, user.name) &&
               Objects.equals(createdAt, user.createdAt);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, email, name, createdAt);
    }

    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", email='" + email + '\'' +
                ", name='" + name + '\'' +
                ", createdAt=" + createdAt +
                '}';
    }

    // Getters
    public long getId() { return id; }
    public String getEmail() { return email; }
    public String getName() { return name; }
    public Date getCreatedAt() { return new Date(createdAt.getTime()); } // Defensive copy
}
```

### Comparable Implementation
```java
public class Message implements Comparable<Message> {
    private final long id;
    private final String content;
    private final long timestamp;
    private final MessagePriority priority;

    public enum MessagePriority {
        LOW(1), NORMAL(2), HIGH(3), URGENT(4);
        
        private final int value;
        
        MessagePriority(int value) {
            this.value = value;
        }
        
        public int getValue() {
            return value;
        }
    }

    public Message(long id, String content, long timestamp, MessagePriority priority) {
        this.id = id;
        this.content = Objects.requireNonNull(content, "Content cannot be null");
        this.timestamp = timestamp;
        this.priority = Objects.requireNonNull(priority, "Priority cannot be null");
    }

    @Override
    public int compareTo(Message other) {
        // First compare by priority (higher priority first)
        int priorityComparison = Integer.compare(other.priority.getValue(), this.priority.getValue());
        if (priorityComparison != 0) {
            return priorityComparison;
        }
        
        // Then compare by timestamp (newer first)
        return Long.compare(other.timestamp, this.timestamp);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        Message message = (Message) obj;
        return id == message.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    // Getters
    public long getId() { return id; }
    public String getContent() { return content; }
    public long getTimestamp() { return timestamp; }
    public MessagePriority getPriority() { return priority; }
}
```

## Classes and Interfaces

### Composition over Inheritance
```java
// Instead of inheritance, use composition
public class SmartImageView extends AppCompatImageView {
    private final ImageLoader imageLoader;
    private final ImageCache imageCache;
    private final ImageTransformer transformer;

    public SmartImageView(Context context) {
        this(context, null);
    }

    public SmartImageView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public SmartImageView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        
        // Compose functionality rather than inherit
        this.imageLoader = new ImageLoader(context);
        this.imageCache = new ImageCache(context);
        this.transformer = new ImageTransformer();
    }

    public void loadImage(String url) {
        loadImage(url, null);
    }

    public void loadImage(String url, ImageLoadCallback callback) {
        if (TextUtils.isEmpty(url)) {
            setImageResource(R.drawable.placeholder);
            return;
        }

        // Check cache first
        Bitmap cached = imageCache.get(url);
        if (cached != null) {
            setImageBitmap(cached);
            if (callback != null) {
                callback.onImageLoaded(cached);
            }
            return;
        }

        // Show placeholder
        setImageResource(R.drawable.placeholder);

        // Load image
        imageLoader.load(url, new ImageLoader.Callback() {
            @Override
            public void onSuccess(Bitmap bitmap) {
                Bitmap transformed = transformer.transform(bitmap);
                imageCache.put(url, transformed);
                setImageBitmap(transformed);
                
                if (callback != null) {
                    callback.onImageLoaded(transformed);
                }
            }

            @Override
            public void onError(Exception error) {
                setImageResource(R.drawable.error);
                if (callback != null) {
                    callback.onImageLoadFailed(error);
                }
            }
        });
    }

    public interface ImageLoadCallback {
        void onImageLoaded(Bitmap bitmap);
        void onImageLoadFailed(Exception error);
    }
}

// Separate concerns into focused classes
public class ImageLoader {
    private final Context context;
    private final ExecutorService executor;

    public ImageLoader(Context context) {
        this.context = context.getApplicationContext();
        this.executor = Executors.newFixedThreadPool(4);
    }

    public void load(String url, Callback callback) {
        executor.execute(() -> {
            try {
                Bitmap bitmap = downloadImage(url);
                Handler mainHandler = new Handler(Looper.getMainLooper());
                mainHandler.post(() -> callback.onSuccess(bitmap));
            } catch (Exception e) {
                Handler mainHandler = new Handler(Looper.getMainLooper());
                mainHandler.post(() -> callback.onError(e));
            }
        });
    }

    private Bitmap downloadImage(String url) throws IOException {
        // Implementation for downloading image
        return null;
    }

    public interface Callback {
        void onSuccess(Bitmap bitmap);
        void onError(Exception error);
    }
}
```

### Interface Segregation
```java
// Instead of one large interface
public interface MediaPlayerActions {
    void play();
    void pause();
    void stop();
    void seekTo(int position);
    void setVolume(float volume);
    void setPlaybackSpeed(float speed);
    void setEqualizer(EqualizerSettings settings);
    void enableSubtitles(boolean enable);
    void setSubtitleTrack(int trackIndex);
}

// Split into focused interfaces
public interface BasicMediaControls {
    void play();
    void pause();
    void stop();
    void seekTo(int position);
}

public interface AudioControls {
    void setVolume(float volume);
    void setPlaybackSpeed(float speed);
    void setEqualizer(EqualizerSettings settings);
}

public interface SubtitleControls {
    void enableSubtitles(boolean enable);
    void setSubtitleTrack(int trackIndex);
    List<SubtitleTrack> getAvailableSubtitleTracks();
}

// Implementation can choose which interfaces to support
public class AudioPlayer implements BasicMediaControls, AudioControls {
    
    @Override
    public void play() {
        // Audio player implementation
    }
    
    @Override
    public void pause() {
        // Audio player implementation
    }
    
    @Override
    public void setVolume(float volume) {
        // Audio player implementation
    }
    
    // Subtitle controls not implemented for audio player
}

public class VideoPlayer implements BasicMediaControls, AudioControls, SubtitleControls {
    
    @Override
    public void play() {
        // Video player implementation
    }
    
    @Override
    public void enableSubtitles(boolean enable) {
        // Video player implementation
    }
    
    // All interfaces implemented for video player
}
```

## Generics and Collections

### Type-Safe Collections
```java
public class Repository<T> {
    private final List<T> items;
    private final Class<T> type;

    public Repository(Class<T> type) {
        this.type = Objects.requireNonNull(type, "Type cannot be null");
        this.items = new ArrayList<>();
    }

    public void add(T item) {
        if (item != null && type.isInstance(item)) {
            items.add(item);
        } else {
            throw new IllegalArgumentException("Item must be of type " + type.getSimpleName());
        }
    }

    public List<T> getAll() {
        return new ArrayList<>(items); // Defensive copy
    }

    public Optional<T> findById(long id) {
        return items.stream()
                .filter(item -> {
                    try {
                        Field idField = type.getDeclaredField("id");
                        idField.setAccessible(true);
                        return Objects.equals(idField.get(item), id);
                    } catch (Exception e) {
                        return false;
                    }
                })
                .findFirst();
    }

    public List<T> findByPredicate(Predicate<T> predicate) {
        return items.stream()
                .filter(predicate)
                .collect(Collectors.toList());
    }

    public boolean remove(T item) {
        return items.remove(item);
    }

    public void clear() {
        items.clear();
    }

    public int size() {
        return items.size();
    }

    public boolean isEmpty() {
        return items.isEmpty();
    }
}

// Usage
public class UserRepository extends Repository<User> {
    
    public UserRepository() {
        super(User.class);
    }
    
    public List<User> findByEmail(String email) {
        return findByPredicate(user -> user.getEmail().equals(email));
    }
    
    public List<User> findActiveUsers() {
        return findByPredicate(user -> user.isActive());
    }
}
```

### Proper Use of Wildcards
```java
public class DataProcessor {
    
    // Producer extends, Consumer super (PECS principle)
    
    // Method that reads from collection (producer)
    public static double calculateAverage(List<? extends Number> numbers) {
        if (numbers.isEmpty()) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (Number num : numbers) {
            sum += num.doubleValue();
        }
        return sum / numbers.size();
    }
    
    // Method that writes to collection (consumer)
    public static void addNumbers(List<? super Integer> numbers, int... values) {
        for (int value : values) {
            numbers.add(value);
        }
    }
    
    // Generic method with bounds
    public static <T extends Comparable<T>> T findMax(List<T> list) {
        if (list.isEmpty()) {
            throw new IllegalArgumentException("List cannot be empty");
        }
        
        T max = list.get(0);
        for (T item : list) {
            if (item.compareTo(max) > 0) {
                max = item;
            }
        }
        return max;
    }
}

// Usage in Android
public class StatisticsHelper {
    
    public void calculateStats(List<User> users) {
        // Extract ages
        List<Integer> ages = users.stream()
                .map(User::getAge)
                .collect(Collectors.toList());
        
        // Calculate average age using generic method
        double averageAge = DataProcessor.calculateAverage(ages);
        
        // Find oldest user
        User oldestUser = DataProcessor.findMax(users);
        
        Log.d("Stats", "Average age: " + averageAge);
        Log.d("Stats", "Oldest user: " + oldestUser.getName());
    }
}
```

## Enums and Annotations

### Rich Enum Implementation
```java
public enum NetworkState {
    CONNECTED(true, "Connected to network"),
    CONNECTING(false, "Connecting to network"),
    DISCONNECTED(false, "Disconnected from network"),
    NO_INTERNET(false, "No internet connection"),
    LIMITED_CONNECTIVITY(false, "Limited connectivity");

    private final boolean isOnline;
    private final String description;

    NetworkState(boolean isOnline, String description) {
        this.isOnline = isOnline;
        this.description = description;
    }

    public boolean isOnline() {
        return isOnline;
    }

    public String getDescription() {
        return description;
    }

    public boolean canMakeNetworkRequests() {
        return this == CONNECTED;
    }

    public boolean shouldRetryConnection() {
        return this == DISCONNECTED || this == NO_INTERNET;
    }

    // Strategy pattern within enum
    public abstract static class NetworkAction {
        public abstract void execute(Context context);
    }

    public NetworkAction getRecommendedAction(Context context) {
        switch (this) {
            case CONNECTED:
                return new NetworkAction() {
                    @Override
                    public void execute(Context context) {
                        // No action needed
                    }
                };
            case DISCONNECTED:
            case NO_INTERNET:
                return new NetworkAction() {
                    @Override
                    public void execute(Context context) {
                        Toast.makeText(context, "Please check your internet connection", 
                            Toast.LENGTH_SHORT).show();
                    }
                };
            case CONNECTING:
                return new NetworkAction() {
                    @Override
                    public void execute(Context context) {
                        // Show loading indicator
                    }
                };
            default:
                return new NetworkAction() {
                    @Override
                    public void execute(Context context) {
                        // Default action
                    }
                };
        }
    }
}
```

### Custom Annotations for Android
```java
// Thread safety annotation
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.METHOD, ElementType.FIELD, ElementType.TYPE})
public @interface ThreadSafe {
    String value() default "";
}

// Background thread annotation
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.METHOD)
public @interface BackgroundThread {
}

// Main thread annotation
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.METHOD)
public @interface MainThread {
}

// Nullable/NonNull with custom messages
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.PARAMETER, ElementType.FIELD, ElementType.METHOD, ElementType.LOCAL_VARIABLE})
public @interface Nullable {
    String message() default "This value can be null";
}

@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.PARAMETER, ElementType.FIELD, ElementType.METHOD, ElementType.LOCAL_VARIABLE})
public @interface NonNull {
    String message() default "This value must not be null";
}

// Usage in Android code
public class DatabaseManager {
    
    @ThreadSafe("Uses synchronized methods")
    private final Map<String, Object> cache = new ConcurrentHashMap<>();
    
    @BackgroundThread
    public List<User> loadUsers(@NonNull String query) {
        if (query == null) {
            throw new IllegalArgumentException("Query cannot be null");
        }
        
        return performDatabaseQuery(query);
    }
    
    @MainThread
    public void updateUI(@Nullable List<User> users) {
        if (users != null) {
            // Update UI components
        }
    }
    
    @ThreadSafe
    public synchronized void cacheResult(@NonNull String key, @NonNull Object value) {
        cache.put(key, value);
    }
}
```

## Lambdas and Streams

### Effective Lambda Usage in Android
```java
public class UserService {
    private final List<User> users;

    public UserService(List<User> users) {
        this.users = new ArrayList<>(users);
    }

    // Filter active users
    public List<User> getActiveUsers() {
        return users.stream()
                .filter(User::isActive)
                .collect(Collectors.toList());
    }

    // Group users by department
    public Map<String, List<User>> getUsersByDepartment() {
        return users.stream()
                .collect(Collectors.groupingBy(User::getDepartment));
    }

    // Find user with highest score
    public Optional<User> getTopPerformer() {
        return users.stream()
                .filter(User::isActive)
                .max(Comparator.comparing(User::getScore));
    }

    // Calculate average score by department
    public Map<String, Double> getAverageScoreByDepartment() {
        return users.stream()
                .filter(User::isActive)
                .collect(Collectors.groupingBy(
                    User::getDepartment,
                    Collectors.averagingDouble(User::getScore)
                ));
    }

    // Get user names sorted alphabetically
    public List<String> getSortedUserNames() {
        return users.stream()
                .map(User::getName)
                .sorted()
                .collect(Collectors.toList());
    }

    // Check if any user has admin privileges
    public boolean hasAdminUsers() {
        return users.stream()
                .anyMatch(User::isAdmin);
    }

    // Convert to display format
    public List<String> getUserDisplayNames() {
        return users.stream()
                .map(user -> String.format("%s (%s)", user.getName(), user.getEmail()))
                .collect(Collectors.toList());
    }
}

// Usage in Android Activity
public class UserListActivity extends AppCompatActivity {
    
    private UserService userService;
    private RecyclerView recyclerView;
    private UserAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_list);

        setupViews();
        loadUsers();
    }

    private void setupViews() {
        recyclerView = findViewById(R.id.recyclerView);
        adapter = new UserAdapter();
        recyclerView.setAdapter(adapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // Setup filter buttons with lambdas
        Button activeUsersButton = findViewById(R.id.activeUsersButton);
        Button allUsersButton = findViewById(R.id.allUsersButton);
        Button adminUsersButton = findViewById(R.id.adminUsersButton);

        activeUsersButton.setOnClickListener(v -> showActiveUsers());
        allUsersButton.setOnClickListener(v -> showAllUsers());
        adminUsersButton.setOnClickListener(v -> showAdminUsers());
    }

    private void showActiveUsers() {
        List<User> activeUsers = userService.getActiveUsers();
        adapter.updateUsers(activeUsers);
    }

    private void showAllUsers() {
        adapter.updateUsers(userService.getAllUsers());
    }

    private void showAdminUsers() {
        List<User> adminUsers = userService.getUsers()
                .stream()
                .filter(User::isAdmin)
                .collect(Collectors.toList());
        adapter.updateUsers(adminUsers);
    }
}
```

## Concurrency in Android

### Proper Thread Management
```java
public class TaskExecutor {
    private final ExecutorService backgroundExecutor;
    private final Handler mainHandler;

    public TaskExecutor() {
        // Create background thread pool
        this.backgroundExecutor = Executors.newFixedThreadPool(
            Runtime.getRuntime().availableProcessors(),
            r -> {
                Thread thread = new Thread(r);
                thread.setName("TaskExecutor-" + thread.getId());
                thread.setDaemon(true);
                return thread;
            }
        );
        
        this.mainHandler = new Handler(Looper.getMainLooper());
    }

    public <T> void executeAsync(Supplier<T> backgroundTask, Consumer<T> onSuccess, Consumer<Exception> onError) {
        backgroundExecutor.execute(() -> {
            try {
                T result = backgroundTask.get();
                mainHandler.post(() -> onSuccess.accept(result));
            } catch (Exception e) {
                mainHandler.post(() -> onError.accept(e));
            }
        });
    }

    public void executeAsync(Runnable backgroundTask, Runnable onComplete) {
        backgroundExecutor.execute(() -> {
            try {
                backgroundTask.run();
                mainHandler.post(onComplete);
            } catch (Exception e) {
                mainHandler.post(() -> Log.e("TaskExecutor", "Background task failed", e));
            }
        });
    }

    public void shutdown() {
        backgroundExecutor.shutdown();
        try {
            if (!backgroundExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                backgroundExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            backgroundExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    // Functional interfaces for Java 7 compatibility
    public interface Supplier<T> {
        T get() throws Exception;
    }

    public interface Consumer<T> {
        void accept(T value);
    }
}

// Usage example
public class DataRepository {
    private final TaskExecutor executor = new TaskExecutor();
    private final ApiService apiService;

    public void loadUserData(String userId, UserDataCallback callback) {
        executor.executeAsync(
            // Background task
            () -> apiService.fetchUser(userId),
            // Success callback (runs on main thread)
            user -> callback.onSuccess(user),
            // Error callback (runs on main thread)
            error -> callback.onError(error)
        );
    }

    public interface UserDataCallback {
        void onSuccess(User user);
        void onError(Exception error);
    }
}
```

## Android-Specific Best Practices

### Resource Management
```java
public class ResourceManager implements Closeable {
    private final List<Closeable> resources = new ArrayList<>();
    private boolean closed = false;

    public <T extends Closeable> T manage(T resource) {
        if (closed) {
            throw new IllegalStateException("ResourceManager is closed");
        }
        resources.add(resource);
        return resource;
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        
        closed = true;
        
        // Close resources in reverse order
        IOException firstException = null;
        for (int i = resources.size() - 1; i >= 0; i--) {
            try {
                resources.get(i).close();
            } catch (IOException e) {
                if (firstException == null) {
                    firstException = e;
                } else {
                    firstException.addSuppressed(e);
                }
            }
        }
        
        resources.clear();
        
        if (firstException != null) {
            throw firstException;
        }
    }
}

// Usage in Android
public class DatabaseActivity extends AppCompatActivity {
    
    private ResourceManager resourceManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_database);

        resourceManager = new ResourceManager();
        setupDatabase();
    }

    private void setupDatabase() {
        try {
            // Manage database resources
            SQLiteDatabase db = resourceManager.manage(
                SQLiteDatabase.openDatabase("path/to/db", null, SQLiteDatabase.OPEN_READONLY)
            );
            
            Cursor cursor = resourceManager.manage(
                db.rawQuery("SELECT * FROM users", null)
            );
            
            // Use resources...
            
        } catch (Exception e) {
            Log.e("DatabaseActivity", "Error setting up database", e);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        if (resourceManager != null) {
            try {
                resourceManager.close();
            } catch (IOException e) {
                Log.e("DatabaseActivity", "Error closing resources", e);
            }
        }
    }
}
```

Understanding and applying effective Java principles in Android development leads to more maintainable, efficient, and robust applications. These patterns help manage complexity, improve performance, and create code that's easier to test and debug.
