# Android Logging - Complete Guide

## Table of Contents
- [Overview](#overview)
- [Android Log System](#android-log-system)
- [Log Levels](#log-levels)
- [Basic Logging](#basic-logging)
- [Advanced Logging Techniques](#advanced-logging-techniques)
- [Custom Logging Solutions](#custom-logging-solutions)
- [Third-Party Logging Libraries](#third-party-logging-libraries)
- [Logging Best Practices](#logging-best-practices)
- [Performance Considerations](#performance-considerations)
- [Debugging with Logs](#debugging-with-logs)
- [Production Logging](#production-logging)

## Overview

Logging is essential for debugging, monitoring, and understanding application behavior. Android provides built-in logging capabilities through the `Log` class and `logcat` tool.

### Key Benefits
- **Debugging**: Track application flow and identify issues
- **Monitoring**: Monitor app performance and user behavior
- **Crash Analysis**: Understand crash causes and conditions
- **Performance**: Identify bottlenecks and optimization opportunities

## Android Log System

### Logcat
```bash
# View all logs
adb logcat

# Filter by tag
adb logcat MyApp:D *:S

# Filter by log level
adb logcat *:W

# Filter by package
adb logcat --pid=$(adb shell pidof com.example.myapp)

# Clear logs
adb logcat -c

# Save logs to file
adb logcat > logs.txt

# View logs with timestamp
adb logcat -v time

# View logs with thread info
adb logcat -v threadtime
```

### Log Buffer Types
```bash
# Main buffer (default)
adb logcat -b main

# System buffer
adb logcat -b system

# Radio buffer
adb logcat -b radio

# Events buffer
adb logcat -b events

# Crash buffer
adb logcat -b crash

# All buffers
adb logcat -b all
```

## Log Levels

### Priority Levels (Lowest to Highest)
```java
public class LogLevels {
    
    public void demonstrateLogLevels() {
        String tag = "LogLevels";
        
        // VERBOSE (2) - Detailed information, typically only of interest when diagnosing problems
        Log.v(tag, "This is a verbose message");
        
        // DEBUG (3) - Debug information that may be useful during development
        Log.d(tag, "This is a debug message");
        
        // INFO (4) - General information messages
        Log.i(tag, "This is an info message");
        
        // WARN (5) - Warning messages for potentially harmful situations
        Log.w(tag, "This is a warning message");
        
        // ERROR (6) - Error messages for serious problems
        Log.e(tag, "This is an error message");
        
        // ASSERT (7) - What a terrible failure! Should never happen
        Log.wtf(tag, "What a Terrible Failure!");
    }
    
    // Log level constants
    public static final int VERBOSE = Log.VERBOSE; // 2
    public static final int DEBUG = Log.DEBUG;     // 3
    public static final int INFO = Log.INFO;       // 4
    public static final int WARN = Log.WARN;       // 5
    public static final int ERROR = Log.ERROR;     // 6
    public static final int ASSERT = Log.ASSERT;   // 7
}
```

### Checking Log Levels
```java
public class LogLevelChecker {
    
    private static final String TAG = "LogLevelChecker";
    
    public void conditionalLogging() {
        // Check if debug logging is enabled
        if (Log.isLoggable(TAG, Log.DEBUG)) {
            Log.d(TAG, "Debug logging is enabled");
        }
        
        // Check if verbose logging is enabled
        if (Log.isLoggable(TAG, Log.VERBOSE)) {
            Log.v(TAG, "Verbose logging is enabled");
        }
        
        // More efficient for expensive operations
        if (BuildConfig.DEBUG) {
            String expensiveDebugInfo = generateExpensiveDebugInfo();
            Log.d(TAG, "Debug info: " + expensiveDebugInfo);
        }
    }
    
    private String generateExpensiveDebugInfo() {
        // Expensive operation only in debug builds
        return "Expensive debug information";
    }
}
```

## Basic Logging

### Simple Logging Examples
```java
public class BasicLogging {
    
    private static final String TAG = "BasicLogging";
    
    public void basicLogExamples() {
        // Simple string logging
        Log.d(TAG, "User clicked the button");
        
        // Logging with variables
        String username = "john_doe";
        int userId = 12345;
        Log.i(TAG, "User logged in: " + username + " (ID: " + userId + ")");
        
        // String formatting
        Log.i(TAG, String.format("User %s (ID: %d) logged in", username, userId));
        
        // Logging with exception
        try {
            int result = 10 / 0;
        } catch (ArithmeticException e) {
            Log.e(TAG, "Division by zero error", e);
        }
        
        // Logging object state
        User user = new User("John", "john@example.com");
        Log.d(TAG, "User object: " + user.toString());
    }
    
    public void logMethodEntry() {
        Log.d(TAG, "Entering logMethodEntry()");
        
        // Method logic here
        
        Log.d(TAG, "Exiting logMethodEntry()");
    }
    
    public void logWithParameters(String param1, int param2) {
        Log.d(TAG, String.format("Method called with params: %s, %d", param1, param2));
    }
}
```

### Application Lifecycle Logging
```java
public class LoggingActivity extends AppCompatActivity {
    
    private static final String TAG = "LoggingActivity";
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate() called");
        
        if (savedInstanceState != null) {
            Log.d(TAG, "Restoring from saved state");
        } else {
            Log.d(TAG, "Creating fresh instance");
        }
        
        setContentView(R.layout.activity_main);
    }
    
    @Override
    protected void onStart() {
        super.onStart();
        Log.d(TAG, "onStart() called");
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "onResume() called");
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        Log.d(TAG, "onPause() called");
    }
    
    @Override
    protected void onStop() {
        super.onStop();
        Log.d(TAG, "onStop() called");
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "onDestroy() called");
    }
    
    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        Log.d(TAG, "onSaveInstanceState() called");
    }
    
    @Override
    protected void onRestoreInstanceState(@NonNull Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        Log.d(TAG, "onRestoreInstanceState() called");
    }
}
```

## Advanced Logging Techniques

### Custom Log Wrapper
```java
public class Logger {
    
    private static final boolean DEBUG = BuildConfig.DEBUG;
    private static final String APP_TAG = "MyApp";
    
    public static void v(String tag, String message) {
        if (DEBUG) {
            Log.v(formatTag(tag), message);
        }
    }
    
    public static void d(String tag, String message) {
        if (DEBUG) {
            Log.d(formatTag(tag), message);
        }
    }
    
    public static void i(String tag, String message) {
        Log.i(formatTag(tag), message);
    }
    
    public static void w(String tag, String message) {
        Log.w(formatTag(tag), message);
    }
    
    public static void e(String tag, String message) {
        Log.e(formatTag(tag), message);
    }
    
    public static void e(String tag, String message, Throwable throwable) {
        Log.e(formatTag(tag), message, throwable);
    }
    
    public static void wtf(String tag, String message) {
        Log.wtf(formatTag(tag), message);
    }
    
    // Method entry/exit logging
    public static void enter(String tag, String methodName) {
        if (DEBUG) {
            Log.d(formatTag(tag), "→ " + methodName + "()");
        }
    }
    
    public static void exit(String tag, String methodName) {
        if (DEBUG) {
            Log.d(formatTag(tag), "← " + methodName + "()");
        }
    }
    
    // Performance logging
    public static void logPerformance(String tag, String operation, long durationMs) {
        Log.i(formatTag(tag), String.format("Performance: %s took %dms", operation, durationMs));
    }
    
    // Network logging
    public static void logNetworkRequest(String tag, String url, String method) {
        Log.d(formatTag(tag), String.format("Network: %s %s", method, url));
    }
    
    public static void logNetworkResponse(String tag, String url, int statusCode, long durationMs) {
        Log.d(formatTag(tag), String.format("Network: %s responded with %d in %dms", url, statusCode, durationMs));
    }
    
    private static String formatTag(String tag) {
        return APP_TAG + "/" + tag;
    }
    
    // Stack trace logging
    public static void logStackTrace(String tag) {
        if (DEBUG) {
            StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
            StringBuilder sb = new StringBuilder();
            sb.append("Stack trace:\n");
            
            // Skip first few elements (getStackTrace, logStackTrace, etc.)
            for (int i = 3; i < Math.min(stackTrace.length, 10); i++) {
                sb.append("  at ").append(stackTrace[i].toString()).append("\n");
            }
            
            Log.d(formatTag(tag), sb.toString());
        }
    }
}

// Usage examples
public class LoggerUsage {
    
    private static final String TAG = "LoggerUsage";
    
    public void demonstrateUsage() {
        Logger.enter(TAG, "demonstrateUsage");
        
        Logger.d(TAG, "This is a debug message");
        Logger.i(TAG, "User performed action");
        
        try {
            performNetworkRequest();
        } catch (Exception e) {
            Logger.e(TAG, "Network request failed", e);
        }
        
        Logger.exit(TAG, "demonstrateUsage");
    }
    
    private void performNetworkRequest() {
        String url = "https://api.example.com/data";
        Logger.logNetworkRequest(TAG, url, "GET");
        
        long startTime = System.currentTimeMillis();
        
        // Simulate network request
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        long duration = System.currentTimeMillis() - startTime;
        Logger.logNetworkResponse(TAG, url, 200, duration);
    }
}
```

### Structured Logging
```java
public class StructuredLogger {
    
    private static final String TAG = "StructuredLogger";
    
    public static class LogEvent {
        private final String event;
        private final Map<String, Object> data;
        private final long timestamp;
        
        public LogEvent(String event) {
            this.event = event;
            this.data = new HashMap<>();
            this.timestamp = System.currentTimeMillis();
        }
        
        public LogEvent put(String key, Object value) {
            data.put(key, value);
            return this;
        }
        
        public void log() {
            try {
                JSONObject json = new JSONObject();
                json.put("event", event);
                json.put("timestamp", timestamp);
                json.put("thread", Thread.currentThread().getName());
                
                JSONObject dataJson = new JSONObject();
                for (Map.Entry<String, Object> entry : data.entrySet()) {
                    dataJson.put(entry.getKey(), entry.getValue());
                }
                json.put("data", dataJson);
                
                Log.i(TAG, json.toString());
                
            } catch (JSONException e) {
                Log.e(TAG, "Failed to create structured log", e);
            }
        }
    }
    
    // User events
    public static void logUserAction(String action, String screen) {
        new LogEvent("user_action")
            .put("action", action)
            .put("screen", screen)
            .put("user_id", getCurrentUserId())
            .log();
    }
    
    // Performance events
    public static void logPerformanceEvent(String operation, long duration) {
        new LogEvent("performance")
            .put("operation", operation)
            .put("duration_ms", duration)
            .put("memory_used", getMemoryUsage())
            .log();
    }
    
    // Error events
    public static void logError(String error, Throwable throwable) {
        new LogEvent("error")
            .put("error_message", error)
            .put("exception_type", throwable.getClass().getSimpleName())
            .put("stack_trace", Log.getStackTraceString(throwable))
            .log();
    }
    
    // Business events
    public static void logBusinessEvent(String eventType, Map<String, Object> properties) {
        LogEvent event = new LogEvent("business_event")
            .put("event_type", eventType);
        
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            event.put(entry.getKey(), entry.getValue());
        }
        
        event.log();
    }
    
    private static String getCurrentUserId() {
        // Get current user ID from your user management system
        return "user_123";
    }
    
    private static long getMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }
}

// Usage examples
public class StructuredLoggingUsage {
    
    public void logUserInteraction() {
        StructuredLogger.logUserAction("button_click", "main_screen");
        
        Map<String, Object> properties = new HashMap<>();
        properties.put("product_id", "12345");
        properties.put("price", 29.99);
        properties.put("category", "electronics");
        
        StructuredLogger.logBusinessEvent("product_view", properties);
    }
    
    public void measurePerformance() {
        long startTime = System.currentTimeMillis();
        
        // Perform operation
        performExpensiveOperation();
        
        long duration = System.currentTimeMillis() - startTime;
        StructuredLogger.logPerformanceEvent("expensive_operation", duration);
    }
    
    private void performExpensiveOperation() {
        // Simulate expensive operation
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

## Custom Logging Solutions

### File Logging
```java
public class FileLogger {
    
    private static final String TAG = "FileLogger";
    private static final String LOG_FILE_NAME = "app_logs.txt";
    private static final int MAX_LOG_FILE_SIZE = 1024 * 1024; // 1MB
    
    private File logFile;
    private PrintWriter writer;
    private final Object lock = new Object();
    
    public FileLogger(Context context) {
        try {
            logFile = new File(context.getFilesDir(), LOG_FILE_NAME);
            initializeWriter();
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize file logger", e);
        }
    }
    
    private void initializeWriter() throws IOException {
        // Rotate log file if it's too large
        if (logFile.exists() && logFile.length() > MAX_LOG_FILE_SIZE) {
            rotateLogFile();
        }
        
        writer = new PrintWriter(new FileWriter(logFile, true));
    }
    
    private void rotateLogFile() {
        File oldLogFile = new File(logFile.getParent(), LOG_FILE_NAME + ".old");
        if (oldLogFile.exists()) {
            oldLogFile.delete();
        }
        logFile.renameTo(oldLogFile);
    }
    
    public void log(int level, String tag, String message) {
        log(level, tag, message, null);
    }
    
    public void log(int level, String tag, String message, Throwable throwable) {
        synchronized (lock) {
            if (writer != null) {
                try {
                    String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
                        .format(new Date());
                    String levelStr = getLevelString(level);
                    String threadName = Thread.currentThread().getName();
                    
                    writer.printf("[%s] %s/%s(%s): %s%n", 
                        timestamp, levelStr, tag, threadName, message);
                    
                    if (throwable != null) {
                        throwable.printStackTrace(writer);
                    }
                    
                    writer.flush();
                    
                    // Also log to system log
                    Log.println(level, tag, message);
                    if (throwable != null) {
                        Log.println(level, tag, Log.getStackTraceString(throwable));
                    }
                    
                } catch (Exception e) {
                    Log.e(TAG, "Failed to write to log file", e);
                }
            }
        }
    }
    
    private String getLevelString(int level) {
        switch (level) {
            case Log.VERBOSE: return "V";
            case Log.DEBUG: return "D";
            case Log.INFO: return "I";
            case Log.WARN: return "W";
            case Log.ERROR: return "E";
            case Log.ASSERT: return "A";
            default: return "?";
        }
    }
    
    public void close() {
        synchronized (lock) {
            if (writer != null) {
                writer.close();
                writer = null;
            }
        }
    }
    
    public File getLogFile() {
        return logFile;
    }
    
    public void clearLogs() {
        synchronized (lock) {
            if (logFile.exists()) {
                logFile.delete();
            }
            try {
                initializeWriter();
            } catch (IOException e) {
                Log.e(TAG, "Failed to reinitialize writer after clearing logs", e);
            }
        }
    }
    
    // Convenience methods
    public void d(String tag, String message) {
        log(Log.DEBUG, tag, message);
    }
    
    public void i(String tag, String message) {
        log(Log.INFO, tag, message);
    }
    
    public void w(String tag, String message) {
        log(Log.WARN, tag, message);
    }
    
    public void e(String tag, String message) {
        log(Log.ERROR, tag, message);
    }
    
    public void e(String tag, String message, Throwable throwable) {
        log(Log.ERROR, tag, message, throwable);
    }
}

// Application class setup
public class MyApplication extends Application {
    
    private static FileLogger fileLogger;
    
    @Override
    public void onCreate() {
        super.onCreate();
        
        // Initialize file logger
        fileLogger = new FileLogger(this);
        
        // Set default uncaught exception handler
        Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
            @Override
            public void uncaughtException(@NonNull Thread thread, @NonNull Throwable ex) {
                fileLogger.e("CRASH", "Uncaught exception in thread " + thread.getName(), ex);
                fileLogger.close();
                
                // Call the default handler
                Thread.getDefaultUncaughtExceptionHandler().uncaughtException(thread, ex);
            }
        });
    }
    
    public static FileLogger getFileLogger() {
        return fileLogger;
    }
    
    @Override
    public void onTerminate() {
        super.onTerminate();
        if (fileLogger != null) {
            fileLogger.close();
        }
    }
}
```

### Remote Logging
```java
public class RemoteLogger {
    
    private static final String TAG = "RemoteLogger";
    private static final String LOG_ENDPOINT = "https://your-log-server.com/api/logs";
    
    private final ExecutorService executor;
    private final Queue<LogEntry> logQueue;
    private final Handler mainHandler;
    
    public RemoteLogger() {
        executor = Executors.newSingleThreadExecutor();
        logQueue = new ConcurrentLinkedQueue<>();
        mainHandler = new Handler(Looper.getMainLooper());
        
        // Start background thread to send logs
        startLogSender();
    }
    
    public static class LogEntry {
        public final String level;
        public final String tag;
        public final String message;
        public final String timestamp;
        public final String deviceInfo;
        public final String appVersion;
        
        public LogEntry(String level, String tag, String message) {
            this.level = level;
            this.tag = tag;
            this.message = message;
            this.timestamp = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).format(new Date());
            this.deviceInfo = Build.MODEL + " (" + Build.VERSION.RELEASE + ")";
            this.appVersion = BuildConfig.VERSION_NAME;
        }
        
        public JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("level", level);
            json.put("tag", tag);
            json.put("message", message);
            json.put("timestamp", timestamp);
            json.put("device_info", deviceInfo);
            json.put("app_version", appVersion);
            return json;
        }
    }
    
    public void log(String level, String tag, String message) {
        LogEntry entry = new LogEntry(level, tag, message);
        logQueue.offer(entry);
    }
    
    private void startLogSender() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                while (!Thread.currentThread().isInterrupted()) {
                    try {
                        List<LogEntry> batch = new ArrayList<>();
                        
                        // Collect batch of logs
                        LogEntry entry;
                        while ((entry = logQueue.poll()) != null && batch.size() < 50) {
                            batch.add(entry);
                        }
                        
                        if (!batch.isEmpty()) {
                            sendLogBatch(batch);
                        }
                        
                        Thread.sleep(30000); // Send every 30 seconds
                        
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    } catch (Exception e) {
                        Log.e(TAG, "Error in log sender", e);
                    }
                }
            }
        });
    }
    
    private void sendLogBatch(List<LogEntry> batch) {
        try {
            JSONArray jsonArray = new JSONArray();
            for (LogEntry entry : batch) {
                jsonArray.put(entry.toJson());
            }
            
            JSONObject payload = new JSONObject();
            payload.put("logs", jsonArray);
            
            // Send HTTP request
            URL url = new URL(LOG_ENDPOINT);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            
            OutputStream os = connection.getOutputStream();
            os.write(payload.toString().getBytes());
            os.close();
            
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                Log.d(TAG, "Successfully sent " + batch.size() + " log entries");
            } else {
                Log.w(TAG, "Failed to send logs, response code: " + responseCode);
                // Re-queue the logs for retry
                for (LogEntry entry : batch) {
                    logQueue.offer(entry);
                }
            }
            
            connection.disconnect();
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to send log batch", e);
            // Re-queue the logs for retry
            for (LogEntry entry : batch) {
                logQueue.offer(entry);
            }
        }
    }
    
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    // Convenience methods
    public void d(String tag, String message) {
        log("DEBUG", tag, message);
    }
    
    public void i(String tag, String message) {
        log("INFO", tag, message);
    }
    
    public void w(String tag, String message) {
        log("WARN", tag, message);
    }
    
    public void e(String tag, String message) {
        log("ERROR", tag, message);
    }
}
```

## Third-Party Logging Libraries

### Timber Library
```gradle
// app/build.gradle
dependencies {
    implementation 'com.jakewharton.timber:timber:5.0.1'
}
```

```java
// Application class
public class MyApplication extends Application {
    
    @Override
    public void onCreate() {
        super.onCreate();
        
        if (BuildConfig.DEBUG) {
            // Debug tree for development
            Timber.plant(new Timber.DebugTree());
        } else {
            // Custom tree for production
            Timber.plant(new CrashReportingTree());
        }
    }
    
    // Custom tree for production logging
    private static class CrashReportingTree extends Timber.Tree {
        @Override
        protected void log(int priority, String tag, @NonNull String message, Throwable t) {
            if (priority == Log.VERBOSE || priority == Log.DEBUG) {
                return; // Don't log verbose and debug in production
            }
            
            // Log to crash reporting service
            if (priority == Log.ERROR) {
                // Example: Firebase Crashlytics
                // FirebaseCrashlytics.getInstance().recordException(t != null ? t : new Exception(message));
            }
            
            // Log to file or remote service
            // fileLogger.log(priority, tag, message, t);
        }
    }
}

// Usage examples
public class TimberUsage {
    
    public void demonstrateTimber() {
        // Simple logging (tag is automatically generated from class name)
        Timber.d("Debug message");
        Timber.i("Info message");
        Timber.w("Warning message");
        Timber.e("Error message");
        
        // Logging with formatting
        String username = "john";
        int count = 5;
        Timber.d("User %s has %d items", username, count);
        
        // Logging exceptions
        try {
            int result = 10 / 0;
        } catch (Exception e) {
            Timber.e(e, "Division error occurred");
        }
        
        // Conditional logging with lambda
        Timber.d(() -> "Expensive debug message: " + generateExpensiveString());
        
        // Tagged logging
        Timber.tag("CustomTag").d("Message with custom tag");
    }
    
    private String generateExpensiveString() {
        // Expensive operation only executed if debug logging is enabled
        return "Expensive computation result";
    }
}
```

### SLF4J Android
```gradle
dependencies {
    implementation 'org.slf4j:slf4j-android:1.7.36'
}
```

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SLF4JUsage {
    
    private static final Logger logger = LoggerFactory.getLogger(SLF4JUsage.class);
    
    public void demonstrateSLF4J() {
        // Basic logging
        logger.debug("Debug message");
        logger.info("Info message");
        logger.warn("Warning message");
        logger.error("Error message");
        
        // Parameterized logging
        String username = "john";
        int age = 25;
        logger.info("User {} is {} years old", username, age);
        
        // Exception logging
        try {
            performOperation();
        } catch (Exception e) {
            logger.error("Operation failed for user {}", username, e);
        }
        
        // Marker-based logging
        Marker important = MarkerFactory.getMarker("IMPORTANT");
        logger.info(important, "This is an important message");
    }
    
    private void performOperation() throws Exception {
        throw new RuntimeException("Simulated error");
    }
}
```

## Logging Best Practices

### Performance-Optimized Logging
```java
public class OptimizedLogging {
    
    private static final String TAG = "OptimizedLogging";
    private static final boolean DEBUG = BuildConfig.DEBUG;
    
    public void demonstrateBestPractices() {
        // 1. Use log level checks for expensive operations
        if (Log.isLoggable(TAG, Log.DEBUG)) {
            String expensiveMessage = createExpensiveLogMessage();
            Log.d(TAG, expensiveMessage);
        }
        
        // 2. Use string formatting instead of concatenation
        String userId = "12345";
        String action = "login";
        
        // BAD: String concatenation
        // Log.d(TAG, "User " + userId + " performed " + action + " at " + System.currentTimeMillis());
        
        // GOOD: String formatting
        Log.d(TAG, String.format("User %s performed %s at %d", userId, action, System.currentTimeMillis()));
        
        // 3. Avoid logging in tight loops
        List<String> items = Arrays.asList("item1", "item2", "item3");
        
        // BAD: Logging in loop
        // for (String item : items) {
        //     Log.d(TAG, "Processing item: " + item);
        // }
        
        // GOOD: Batch logging
        if (DEBUG) {
            Log.d(TAG, "Processing items: " + items.toString());
        }
        
        // 4. Use appropriate log levels
        Log.v(TAG, "Verbose debugging info"); // Only for detailed debugging
        Log.d(TAG, "Debug info");             // Development debugging
        Log.i(TAG, "General information");    // Important app flow
        Log.w(TAG, "Warning condition");      // Potential issues
        Log.e(TAG, "Error occurred");         // Actual errors
        
        // 5. Don't log sensitive information
        String password = "secret123";
        String creditCard = "1234-5678-9012-3456";
        
        // BAD: Logging sensitive data
        // Log.d(TAG, "User password: " + password);
        // Log.d(TAG, "Credit card: " + creditCard);
        
        // GOOD: Log without sensitive data
        Log.d(TAG, "User authentication successful");
        Log.d(TAG, "Payment method added (ending in " + creditCard.substring(creditCard.length() - 4) + ")");
    }
    
    private String createExpensiveLogMessage() {
        // Simulate expensive operation
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb.append("data").append(i).append(" ");
        }
        return sb.toString();
    }
}
```

### Structured Logging Best Practices
```java
public class StructuredLoggingBestPractices {
    
    private static final String TAG = "StructuredLogging";
    
    // Use consistent event types
    public static final String EVENT_USER_ACTION = "user_action";
    public static final String EVENT_NETWORK_REQUEST = "network_request";
    public static final String EVENT_PERFORMANCE = "performance";
    public static final String EVENT_ERROR = "error";
    
    // Use consistent field names
    public static final String FIELD_USER_ID = "user_id";
    public static final String FIELD_SESSION_ID = "session_id";
    public static final String FIELD_TIMESTAMP = "timestamp";
    public static final String FIELD_DURATION = "duration_ms";
    public static final String FIELD_URL = "url";
    public static final String FIELD_STATUS_CODE = "status_code";
    
    public void logUserAction(String action, String screen, String userId) {
        Map<String, Object> data = new HashMap<>();
        data.put("action", action);
        data.put("screen", screen);
        data.put(FIELD_USER_ID, userId);
        data.put(FIELD_SESSION_ID, getCurrentSessionId());
        data.put(FIELD_TIMESTAMP, System.currentTimeMillis());
        
        logStructuredEvent(EVENT_USER_ACTION, data);
    }
    
    public void logNetworkRequest(String method, String url, int statusCode, long duration) {
        Map<String, Object> data = new HashMap<>();
        data.put("method", method);
        data.put(FIELD_URL, url);
        data.put(FIELD_STATUS_CODE, statusCode);
        data.put(FIELD_DURATION, duration);
        data.put(FIELD_TIMESTAMP, System.currentTimeMillis());
        
        logStructuredEvent(EVENT_NETWORK_REQUEST, data);
    }
    
    public void logPerformanceMetric(String operation, long duration, Map<String, Object> additionalData) {
        Map<String, Object> data = new HashMap<>();
        data.put("operation", operation);
        data.put(FIELD_DURATION, duration);
        data.put(FIELD_TIMESTAMP, System.currentTimeMillis());
        
        if (additionalData != null) {
            data.putAll(additionalData);
        }
        
        logStructuredEvent(EVENT_PERFORMANCE, data);
    }
    
    private void logStructuredEvent(String eventType, Map<String, Object> data) {
        try {
            JSONObject json = new JSONObject();
            json.put("event_type", eventType);
            
            for (Map.Entry<String, Object> entry : data.entrySet()) {
                json.put(entry.getKey(), entry.getValue());
            }
            
            Log.i(TAG, json.toString());
            
        } catch (JSONException e) {
            Log.e(TAG, "Failed to create structured log for event: " + eventType, e);
        }
    }
    
    private String getCurrentSessionId() {
        // Return current session ID
        return "session_" + System.currentTimeMillis();
    }
}
```

## Performance Considerations

### Log Performance Monitoring
```java
public class LogPerformanceMonitor {
    
    private static final String TAG = "LogPerformance";
    private static final Map<String, Long> operationTimes = new ConcurrentHashMap<>();
    
    public static void startTiming(String operation) {
        operationTimes.put(operation, System.currentTimeMillis());
    }
    
    public static void endTiming(String operation) {
        Long startTime = operationTimes.remove(operation);
        if (startTime != null) {
            long duration = System.currentTimeMillis() - startTime;
            Log.d(TAG, String.format("Operation '%s' took %dms", operation, duration));
            
            // Log slow operations as warnings
            if (duration > 1000) {
                Log.w(TAG, String.format("SLOW OPERATION: '%s' took %dms", operation, duration));
            }
        }
    }
    
    public static void measureOperation(String operation, Runnable runnable) {
        long startTime = System.currentTimeMillis();
        
        try {
            runnable.run();
        } finally {
            long duration = System.currentTimeMillis() - startTime;
            Log.d(TAG, String.format("Operation '%s' completed in %dms", operation, duration));
        }
    }
    
    public static <T> T measureOperation(String operation, Supplier<T> supplier) {
        long startTime = System.currentTimeMillis();
        
        try {
            return supplier.get();
        } finally {
            long duration = System.currentTimeMillis() - startTime;
            Log.d(TAG, String.format("Operation '%s' completed in %dms", operation, duration));
        }
    }
    
    @FunctionalInterface
    public interface Supplier<T> {
        T get();
    }
}

// Usage examples
public class PerformanceLoggingUsage {
    
    public void demonstratePerformanceLogging() {
        // Method 1: Manual timing
        LogPerformanceMonitor.startTiming("database_query");
        performDatabaseQuery();
        LogPerformanceMonitor.endTiming("database_query");
        
        // Method 2: Wrapper timing
        LogPerformanceMonitor.measureOperation("file_processing", () -> {
            processLargeFile();
        });
        
        // Method 3: Timing with return value
        String result = LogPerformanceMonitor.measureOperation("api_call", () -> {
            return callApi();
        });
    }
    
    private void performDatabaseQuery() {
        // Simulate database operation
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    private void processLargeFile() {
        // Simulate file processing
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    private String callApi() {
        // Simulate API call
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return "API response";
    }
}
```

## Debugging with Logs

### Debug Helper Class
```java
public class DebugHelper {
    
    private static final String TAG = "DebugHelper";
    private static final boolean DEBUG = BuildConfig.DEBUG;
    
    // Object state logging
    public static void logObjectState(String tag, Object obj) {
        if (!DEBUG) return;
        
        if (obj == null) {
            Log.d(tag, "Object is null");
            return;
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append(obj.getClass().getSimpleName()).append(" {");
        
        try {
            Field[] fields = obj.getClass().getDeclaredFields();
            for (Field field : fields) {
                field.setAccessible(true);
                Object value = field.get(obj);
                sb.append("\n  ").append(field.getName()).append(" = ").append(value);
            }
        } catch (IllegalAccessException e) {
            sb.append("\n  Error accessing fields: ").append(e.getMessage());
        }
        
        sb.append("\n}");
        Log.d(tag, sb.toString());
    }
    
    // Collection logging
    public static void logCollection(String tag, String name, Collection<?> collection) {
        if (!DEBUG) return;
        
        if (collection == null) {
            Log.d(tag, name + " is null");
            return;
        }
        
        Log.d(tag, String.format("%s (size: %d): %s", name, collection.size(), collection.toString()));
    }
    
    // Map logging
    public static void logMap(String tag, String name, Map<?, ?> map) {
        if (!DEBUG) return;
        
        if (map == null) {
            Log.d(tag, name + " is null");
            return;
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append(name).append(" (size: ").append(map.size()).append(") {");
        
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            sb.append("\n  ").append(entry.getKey()).append(" = ").append(entry.getValue());
        }
        
        sb.append("\n}");
        Log.d(tag, sb.toString());
    }
    
    // Thread information
    public static void logThreadInfo(String tag) {
        if (!DEBUG) return;
        
        Thread thread = Thread.currentThread();
        Log.d(tag, String.format("Thread: %s (ID: %d, Priority: %d, State: %s)",
            thread.getName(), thread.getId(), thread.getPriority(), thread.getState()));
    }
    
    // Memory information
    public static void logMemoryInfo(String tag) {
        if (!DEBUG) return;
        
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        long maxMemory = runtime.maxMemory();
        
        Log.d(tag, String.format("Memory - Used: %d KB, Free: %d KB, Total: %d KB, Max: %d KB",
            usedMemory / 1024, freeMemory / 1024, totalMemory / 1024, maxMemory / 1024));
    }
    
    // Method execution logging
    public static void logMethodExecution(String tag, String methodName, Object... params) {
        if (!DEBUG) return;
        
        StringBuilder sb = new StringBuilder();
        sb.append("→ ").append(methodName).append("(");
        
        if (params.length > 0) {
            for (int i = 0; i < params.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(params[i]);
            }
        }
        
        sb.append(")");
        Log.d(tag, sb.toString());
    }
    
    // Dump stack trace
    public static void logStackTrace(String tag, String message) {
        if (!DEBUG) return;
        
        StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
        StringBuilder sb = new StringBuilder();
        sb.append(message).append("\nStack trace:");
        
        // Skip first few elements (getStackTrace, logStackTrace, etc.)
        for (int i = 3; i < Math.min(stackTrace.length, 15); i++) {
            sb.append("\n  at ").append(stackTrace[i].toString());
        }
        
        Log.d(tag, sb.toString());
    }
}
```

## Production Logging

### Production Logging Strategy
```java
public class ProductionLogger {
    
    private static final String TAG = "ProductionLogger";
    private static final boolean IS_PRODUCTION = !BuildConfig.DEBUG;
    
    private static FileLogger fileLogger;
    private static RemoteLogger remoteLogger;
    
    public static void initialize(Context context) {
        if (IS_PRODUCTION) {
            fileLogger = new FileLogger(context);
            remoteLogger = new RemoteLogger();
        }
    }
    
    public static void logCrash(Throwable throwable) {
        String message = "App crashed: " + throwable.getMessage();
        
        // Always log crashes
        Log.e(TAG, message, throwable);
        
        if (IS_PRODUCTION) {
            if (fileLogger != null) {
                fileLogger.e(TAG, message, throwable);
            }
            
            if (remoteLogger != null) {
                remoteLogger.e(TAG, message + "\n" + Log.getStackTraceString(throwable));
            }
            
            // Send to crash reporting service
            // FirebaseCrashlytics.getInstance().recordException(throwable);
        }
    }
    
    public static void logUserAction(String action, Map<String, String> properties) {
        if (IS_PRODUCTION) {
            // Log important user actions for analytics
            StringBuilder sb = new StringBuilder();
            sb.append("User action: ").append(action);
            
            if (properties != null && !properties.isEmpty()) {
                sb.append(" with properties: ").append(properties.toString());
            }
            
            String message = sb.toString();
            Log.i(TAG, message);
            
            if (remoteLogger != null) {
                remoteLogger.i(TAG, message);
            }
            
            // Send to analytics service
            // FirebaseAnalytics.getInstance(context).logEvent(action, bundle);
        }
    }
    
    public static void logPerformanceIssue(String operation, long duration) {
        if (duration > 5000) { // Log operations taking more than 5 seconds
            String message = String.format("Performance issue: %s took %dms", operation, duration);
            Log.w(TAG, message);
            
            if (IS_PRODUCTION) {
                if (fileLogger != null) {
                    fileLogger.w(TAG, message);
                }
                
                if (remoteLogger != null) {
                    remoteLogger.w(TAG, message);
                }
                
                // Send to performance monitoring service
                // FirebasePerformance.getInstance().newTrace(operation).stop();
            }
        }
    }
    
    public static void logSecurityEvent(String event, String details) {
        String message = String.format("Security event: %s - %s", event, details);
        Log.w(TAG, message);
        
        if (IS_PRODUCTION) {
            if (fileLogger != null) {
                fileLogger.w(TAG, message);
            }
            
            if (remoteLogger != null) {
                remoteLogger.w(TAG, message);
            }
            
            // Send to security monitoring service
        }
    }
    
    public static void shutdown() {
        if (fileLogger != null) {
            fileLogger.close();
        }
        
        if (remoteLogger != null) {
            remoteLogger.shutdown();
        }
    }
}
```

### Log Management and Cleanup
```java
public class LogManager {
    
    private static final String TAG = "LogManager";
    private static final long MAX_LOG_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
    private static final long MAX_TOTAL_LOG_SIZE = 10 * 1024 * 1024; // 10 MB
    
    public static void cleanupOldLogs(Context context) {
        new Thread(() -> {
            try {
                File logsDir = new File(context.getFilesDir(), "logs");
                if (!logsDir.exists()) {
                    return;
                }
                
                File[] logFiles = logsDir.listFiles();
                if (logFiles == null) {
                    return;
                }
                
                long currentTime = System.currentTimeMillis();
                long totalSize = 0;
                
                // Calculate total size and remove old files
                List<File> validFiles = new ArrayList<>();
                for (File file : logFiles) {
                    if (currentTime - file.lastModified() > MAX_LOG_AGE_MS) {
                        file.delete();
                        Log.d(TAG, "Deleted old log file: " + file.getName());
                    } else {
                        validFiles.add(file);
                        totalSize += file.length();
                    }
                }
                
                // If total size is still too large, remove oldest files
                if (totalSize > MAX_TOTAL_LOG_SIZE) {
                    validFiles.sort((f1, f2) -> Long.compare(f1.lastModified(), f2.lastModified()));
                    
                    for (File file : validFiles) {
                        if (totalSize <= MAX_TOTAL_LOG_SIZE) {
                            break;
                        }
                        
                        totalSize -= file.length();
                        file.delete();
                        Log.d(TAG, "Deleted log file to free space: " + file.getName());
                    }
                }
                
                Log.d(TAG, "Log cleanup completed. Remaining size: " + totalSize + " bytes");
                
            } catch (Exception e) {
                Log.e(TAG, "Error during log cleanup", e);
            }
        }).start();
    }
    
    public static void exportLogs(Context context, OnLogExportListener listener) {
        new Thread(() -> {
            try {
                File logsDir = new File(context.getFilesDir(), "logs");
                if (!logsDir.exists()) {
                    listener.onError("No logs directory found");
                    return;
                }
                
                File[] logFiles = logsDir.listFiles();
                if (logFiles == null || logFiles.length == 0) {
                    listener.onError("No log files found");
                    return;
                }
                
                // Create zip file with all logs
                File exportFile = new File(context.getExternalFilesDir(null), "logs_export.zip");
                
                try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(exportFile))) {
                    for (File logFile : logFiles) {
                        ZipEntry entry = new ZipEntry(logFile.getName());
                        zos.putNextEntry(entry);
                        
                        try (FileInputStream fis = new FileInputStream(logFile)) {
                            byte[] buffer = new byte[1024];
                            int length;
                            while ((length = fis.read(buffer)) > 0) {
                                zos.write(buffer, 0, length);
                            }
                        }
                        
                        zos.closeEntry();
                    }
                }
                
                listener.onSuccess(exportFile);
                
            } catch (Exception e) {
                Log.e(TAG, "Error exporting logs", e);
                listener.onError("Export failed: " + e.getMessage());
            }
        }).start();
    }
    
    public interface OnLogExportListener {
        void onSuccess(File exportedFile);
        void onError(String error);
    }
}
```

This comprehensive logging guide covers everything from basic Android logging to advanced production logging strategies. Understanding these concepts will help you effectively debug, monitor, and maintain your Android applications.
