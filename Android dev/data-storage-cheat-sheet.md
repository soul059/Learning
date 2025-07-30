# Android Data Storage Cheat Sheet - Written Exam Reference

## Table of Contents
1. [Storage Options Overview](#storage-options-overview)
2. [SharedPreferences](#sharedpreferences)
3. [Internal Storage](#internal-storage)
4. [External Storage](#external-storage)
5. [SQLite Database](#sqlite-database)
6. [Room Database](#room-database)
7. [Content Providers](#content-providers)
8. [Network Storage](#network-storage)
9. [Storage Permissions](#storage-permissions)
10. [Best Practices](#best-practices)

## Storage Options Overview

### Storage Types Comparison
| Storage Type | Size Limit | Accessibility | Persistence | Use Case |
|--------------|------------|---------------|-------------|----------|
| SharedPreferences | Small | App only | Until uninstall | Settings, flags |
| Internal Storage | Device limit | App only | Until uninstall | Private files |
| External Storage | SD card limit | Other apps | Survives uninstall | Public files |
| SQLite | Device limit | App only | Until uninstall | Structured data |
| Room | Device limit | App only | Until uninstall | Modern database |
| Content Provider | Varies | Other apps | App dependent | Shared data |
| Network | Unlimited | Internet required | Server dependent | Cloud storage |

### When to Use Each Storage Type
```
SharedPreferences: User settings, login status, small key-value pairs
Internal Storage: Private files, cache, temporary data
External Storage: Large files, media, documents to share
SQLite/Room: Complex data relationships, queries
Content Provider: Sharing data between apps
Network: Synchronization, backup, cloud features
```

## SharedPreferences

### Basic Implementation
```java
// Get SharedPreferences
SharedPreferences sharedPref = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE);

// Alternative ways to get SharedPreferences
SharedPreferences defaultPrefs = PreferenceManager.getDefaultSharedPreferences(this);
SharedPreferences activityPrefs = getPreferences(Context.MODE_PRIVATE);
```

### Writing Data
```java
SharedPreferences.Editor editor = sharedPref.edit();

// Store different data types
editor.putString("username", "john_doe");
editor.putInt("user_id", 12345);
editor.putBoolean("is_logged_in", true);
editor.putFloat("rating", 4.5f);
editor.putLong("timestamp", System.currentTimeMillis());

// String Set (API 11+)
Set<String> stringSet = new HashSet<>();
stringSet.add("item1");
stringSet.add("item2");
editor.putStringSet("my_set", stringSet);

// Apply changes
editor.apply(); // Asynchronous
// OR
editor.commit(); // Synchronous, returns boolean
```

### Reading Data
```java
// Read with default values
String username = sharedPref.getString("username", "default_user");
int userId = sharedPref.getInt("user_id", -1);
boolean isLoggedIn = sharedPref.getBoolean("is_logged_in", false);
float rating = sharedPref.getFloat("rating", 0.0f);
long timestamp = sharedPref.getLong("timestamp", 0L);

// Read String Set
Set<String> stringSet = sharedPref.getStringSet("my_set", new HashSet<String>());

// Check if key exists
boolean hasUsername = sharedPref.contains("username");

// Get all keys
Map<String, ?> allEntries = sharedPref.getAll();
```

### Removing Data
```java
SharedPreferences.Editor editor = sharedPref.edit();

// Remove specific key
editor.remove("username");

// Clear all data
editor.clear();

editor.apply();
```

### SharedPreferences Listener
```java
SharedPreferences.OnSharedPreferenceChangeListener listener = 
    new SharedPreferences.OnSharedPreferenceChangeListener() {
    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        if (key.equals("username")) {
            // Handle username change
        }
    }
};

// Register listener
sharedPref.registerOnSharedPreferenceChangeListener(listener);

// Unregister listener (important!)
sharedPref.unregisterOnSharedPreferenceChangeListener(listener);
```

## Internal Storage

### File Operations
```java
// Write to internal storage
String filename = "myfile.txt";
String fileContents = "Hello World!";

try {
    FileOutputStream fos = openFileOutput(filename, Context.MODE_PRIVATE);
    fos.write(fileContents.getBytes());
    fos.close();
} catch (IOException e) {
    e.printStackTrace();
}

// Read from internal storage
try {
    FileInputStream fis = openFileInput(filename);
    InputStreamReader isr = new InputStreamReader(fis);
    BufferedReader br = new BufferedReader(isr);
    StringBuilder sb = new StringBuilder();
    String line;
    while ((line = br.readLine()) != null) {
        sb.append(line);
    }
    br.close();
    String fileContents = sb.toString();
} catch (IOException e) {
    e.printStackTrace();
}
```

### File Modes
```java
Context.MODE_PRIVATE     // Default, file only accessible by your app
Context.MODE_APPEND      // Append to existing file
Context.MODE_WORLD_READABLE   // Deprecated (API 17)
Context.MODE_WORLD_WRITEABLE  // Deprecated (API 17)
```

### Working with Files and Directories
```java
// Get files directory
File filesDir = getFilesDir(); // /data/data/package/files/

// Get cache directory
File cacheDir = getCacheDir(); // /data/data/package/cache/

// Create subdirectory
File customDir = new File(getFilesDir(), "custom");
if (!customDir.exists()) {
    customDir.mkdirs();
}

// Get file in custom directory
File file = new File(customDir, "myfile.txt");

// Check file properties
boolean exists = file.exists();
boolean isDirectory = file.isDirectory();
long size = file.length();
long lastModified = file.lastModified();

// List files
File[] files = filesDir.listFiles();

// Delete file
boolean deleted = file.delete();

// Delete directory (must be empty)
boolean deletedDir = customDir.delete();
```

### Using File Streams
```java
// Write using FileWriter
try {
    File file = new File(getFilesDir(), "example.txt");
    FileWriter writer = new FileWriter(file);
    writer.write("Hello World!");
    writer.close();
} catch (IOException e) {
    e.printStackTrace();
}

// Read using FileReader
try {
    File file = new File(getFilesDir(), "example.txt");
    FileReader reader = new FileReader(file);
    BufferedReader br = new BufferedReader(reader);
    String content = br.readLine();
    br.close();
} catch (IOException e) {
    e.printStackTrace();
}
```

## External Storage

### Checking External Storage State
```java
// Check if external storage is available
String state = Environment.getExternalStorageState();

if (Environment.MEDIA_MOUNTED.equals(state)) {
    // Read and write access
} else if (Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
    // Read-only access
} else {
    // No access
}

// Alternative method
boolean isExternalStorageWritable = Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED);
boolean isExternalStorageReadable = Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED) ||
    Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED_READ_ONLY);
```

### External Storage Paths
```java
// Public external storage (survives app uninstall)
File publicDir = Environment.getExternalStorageDirectory(); // /storage/emulated/0/
File publicDownloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
File publicPictures = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
File publicMusic = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC);

// App-specific external storage (deleted with app)
File externalFilesDir = getExternalFilesDir(null); // /storage/emulated/0/Android/data/package/files/
File externalCacheDir = getExternalCacheDir(); // /storage/emulated/0/Android/data/package/cache/
File externalPicturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
```

### External Storage Operations
```java
// Write to external storage
File file = new File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "myfile.txt");
try {
    FileOutputStream fos = new FileOutputStream(file);
    fos.write("Hello External Storage!".getBytes());
    fos.close();
} catch (IOException e) {
    e.printStackTrace();
}

// Media Scanner (to make files visible in gallery/music apps)
MediaScannerConnection.scanFile(this, new String[]{file.getAbsolutePath()}, null, null);
```

### Scoped Storage (Android 10+)
```java
// Access app-specific directories (no permission needed)
File appSpecificDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);

// Access shared storage using Storage Access Framework
Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
intent.addCategory(Intent.CATEGORY_OPENABLE);
intent.setType("text/plain");
intent.putExtra(Intent.EXTRA_TITLE, "myfile.txt");
startActivityForResult(intent, CREATE_FILE_REQUEST);

// Handle result
@Override
protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    if (requestCode == CREATE_FILE_REQUEST && resultCode == RESULT_OK) {
        Uri uri = data.getData();
        // Write to uri using ContentResolver
        try {
            OutputStream os = getContentResolver().openOutputStream(uri);
            os.write("Hello Scoped Storage!".getBytes());
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

## SQLite Database

### SQLiteOpenHelper Implementation
```java
public class DatabaseHelper extends SQLiteOpenHelper {
    private static final String DATABASE_NAME = "mydatabase.db";
    private static final int DATABASE_VERSION = 1;
    
    // Table and column names
    public static final String TABLE_USERS = "users";
    public static final String COLUMN_ID = "_id";
    public static final String COLUMN_NAME = "name";
    public static final String COLUMN_EMAIL = "email";
    public static final String COLUMN_AGE = "age";
    
    // Create table SQL
    private static final String CREATE_TABLE_USERS = 
        "CREATE TABLE " + TABLE_USERS + " (" +
        COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
        COLUMN_NAME + " TEXT NOT NULL, " +
        COLUMN_EMAIL + " TEXT UNIQUE, " +
        COLUMN_AGE + " INTEGER" +
        ");";
    
    public DatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }
    
    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(CREATE_TABLE_USERS);
    }
    
    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_USERS);
        onCreate(db);
    }
}
```

### Database Operations
```java
DatabaseHelper dbHelper = new DatabaseHelper(this);

// Insert data
SQLiteDatabase db = dbHelper.getWritableDatabase();
ContentValues values = new ContentValues();
values.put(DatabaseHelper.COLUMN_NAME, "John Doe");
values.put(DatabaseHelper.COLUMN_EMAIL, "john@example.com");
values.put(DatabaseHelper.COLUMN_AGE, 25);

long newRowId = db.insert(DatabaseHelper.TABLE_USERS, null, values);

// Query data
String[] projection = {
    DatabaseHelper.COLUMN_ID,
    DatabaseHelper.COLUMN_NAME,
    DatabaseHelper.COLUMN_EMAIL,
    DatabaseHelper.COLUMN_AGE
};

String selection = DatabaseHelper.COLUMN_AGE + " > ?";
String[] selectionArgs = {"18"};
String sortOrder = DatabaseHelper.COLUMN_NAME + " ASC";

Cursor cursor = db.query(
    DatabaseHelper.TABLE_USERS,
    projection,
    selection,
    selectionArgs,
    null,
    null,
    sortOrder
);

// Read cursor
while (cursor.moveToNext()) {
    long id = cursor.getLong(cursor.getColumnIndexOrThrow(DatabaseHelper.COLUMN_ID));
    String name = cursor.getString(cursor.getColumnIndexOrThrow(DatabaseHelper.COLUMN_NAME));
    String email = cursor.getString(cursor.getColumnIndexOrThrow(DatabaseHelper.COLUMN_EMAIL));
    int age = cursor.getInt(cursor.getColumnIndexOrThrow(DatabaseHelper.COLUMN_AGE));
}
cursor.close();

// Update data
ContentValues updateValues = new ContentValues();
updateValues.put(DatabaseHelper.COLUMN_AGE, 26);

String whereClause = DatabaseHelper.COLUMN_ID + " = ?";
String[] whereArgs = {"1"};

int rowsUpdated = db.update(
    DatabaseHelper.TABLE_USERS,
    updateValues,
    whereClause,
    whereArgs
);

// Delete data
String deleteWhereClause = DatabaseHelper.COLUMN_ID + " = ?";
String[] deleteWhereArgs = {"1"};

int rowsDeleted = db.delete(
    DatabaseHelper.TABLE_USERS,
    deleteWhereClause,
    deleteWhereArgs
);

db.close();
```

### Raw SQL Queries
```java
// Raw query
String sql = "SELECT * FROM " + DatabaseHelper.TABLE_USERS + " WHERE " + 
             DatabaseHelper.COLUMN_AGE + " > ?";
Cursor cursor = db.rawQuery(sql, new String[]{"18"});

// Execute SQL
db.execSQL("DELETE FROM " + DatabaseHelper.TABLE_USERS + " WHERE " + 
           DatabaseHelper.COLUMN_AGE + " < 18");

// Prepared statements
SQLiteStatement statement = db.compileStatement(
    "INSERT INTO " + DatabaseHelper.TABLE_USERS + 
    " (name, email, age) VALUES (?, ?, ?)"
);
statement.bindString(1, "Jane Doe");
statement.bindString(2, "jane@example.com");
statement.bindLong(3, 30);
long rowId = statement.executeInsert();
```

### Transactions
```java
SQLiteDatabase db = dbHelper.getWritableDatabase();
db.beginTransaction();
try {
    // Multiple database operations
    ContentValues values1 = new ContentValues();
    values1.put(DatabaseHelper.COLUMN_NAME, "User1");
    db.insert(DatabaseHelper.TABLE_USERS, null, values1);
    
    ContentValues values2 = new ContentValues();
    values2.put(DatabaseHelper.COLUMN_NAME, "User2");
    db.insert(DatabaseHelper.TABLE_USERS, null, values2);
    
    db.setTransactionSuccessful();
} finally {
    db.endTransaction();
}
db.close();
```

## Room Database

### Room Components
```java
// Entity
@Entity(tableName = "users")
public class User {
    @PrimaryKey(autoGenerate = true)
    public int id;
    
    @ColumnInfo(name = "user_name")
    public String name;
    
    @ColumnInfo(name = "user_email")
    public String email;
    
    public int age;
    
    // Constructor, getters, setters
    public User(String name, String email, int age) {
        this.name = name;
        this.email = email;
        this.age = age;
    }
}

// DAO (Data Access Object)
@Dao
public interface UserDao {
    @Query("SELECT * FROM users")
    List<User> getAll();
    
    @Query("SELECT * FROM users WHERE id = :id")
    User getById(int id);
    
    @Query("SELECT * FROM users WHERE age > :minAge")
    List<User> getUsersOlderThan(int minAge);
    
    @Insert
    long insert(User user);
    
    @Insert
    void insertAll(User... users);
    
    @Update
    void update(User user);
    
    @Delete
    void delete(User user);
    
    @Query("DELETE FROM users WHERE id = :id")
    void deleteById(int id);
}

// Database
@Database(entities = {User.class}, version = 1)
public abstract class AppDatabase extends RoomDatabase {
    public abstract UserDao userDao();
    
    private static volatile AppDatabase INSTANCE;
    
    public static AppDatabase getDatabase(final Context context) {
        if (INSTANCE == null) {
            synchronized (AppDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(
                        context.getApplicationContext(),
                        AppDatabase.class,
                        "app_database"
                    ).build();
                }
            }
        }
        return INSTANCE;
    }
}
```

### Room Usage
```java
// Get database instance
AppDatabase db = AppDatabase.getDatabase(this);
UserDao userDao = db.userDao();

// Room operations must be done on background thread
ExecutorService executor = Executors.newSingleThreadExecutor();

// Insert
executor.execute(() -> {
    User user = new User("John Doe", "john@example.com", 25);
    long id = userDao.insert(user);
});

// Query
executor.execute(() -> {
    List<User> users = userDao.getAll();
    runOnUiThread(() -> {
        // Update UI with users
    });
});

// Update
executor.execute(() -> {
    User user = userDao.getById(1);
    user.age = 26;
    userDao.update(user);
});

// Delete
executor.execute(() -> {
    userDao.deleteById(1);
});
```

### Room with LiveData
```java
// DAO with LiveData
@Dao
public interface UserDao {
    @Query("SELECT * FROM users")
    LiveData<List<User>> getAllLive();
    
    @Query("SELECT * FROM users WHERE age > :minAge")
    LiveData<List<User>> getUsersOlderThanLive(int minAge);
}

// Usage in Activity/Fragment
AppDatabase db = AppDatabase.getDatabase(this);
UserDao userDao = db.userDao();

LiveData<List<User>> usersLiveData = userDao.getAllLive();
usersLiveData.observe(this, users -> {
    // Update UI automatically when data changes
});
```

## Content Providers

### Creating a Content Provider
```java
public class MyContentProvider extends ContentProvider {
    private static final String AUTHORITY = "com.example.provider";
    private static final String PATH_USERS = "users";
    private static final String PATH_USER_ID = "users/#";
    
    private static final int USERS = 1;
    private static final int USER_ID = 2;
    
    private static final UriMatcher uriMatcher = new UriMatcher(UriMatcher.NO_MATCH);
    
    static {
        uriMatcher.addURI(AUTHORITY, PATH_USERS, USERS);
        uriMatcher.addURI(AUTHORITY, PATH_USER_ID, USER_ID);
    }
    
    public static final Uri CONTENT_URI = Uri.parse("content://" + AUTHORITY + "/" + PATH_USERS);
    
    private DatabaseHelper dbHelper;
    
    @Override
    public boolean onCreate() {
        dbHelper = new DatabaseHelper(getContext());
        return true;
    }
    
    @Override
    public Cursor query(Uri uri, String[] projection, String selection,
                       String[] selectionArgs, String sortOrder) {
        SQLiteDatabase db = dbHelper.getReadableDatabase();
        Cursor cursor;
        
        switch (uriMatcher.match(uri)) {
            case USERS:
                cursor = db.query(DatabaseHelper.TABLE_USERS, projection, 
                                selection, selectionArgs, null, null, sortOrder);
                break;
            case USER_ID:
                String id = uri.getLastPathSegment();
                cursor = db.query(DatabaseHelper.TABLE_USERS, projection,
                                DatabaseHelper.COLUMN_ID + "=?", new String[]{id},
                                null, null, sortOrder);
                break;
            default:
                throw new IllegalArgumentException("Unknown URI: " + uri);
        }
        
        cursor.setNotificationUri(getContext().getContentResolver(), uri);
        return cursor;
    }
    
    @Override
    public Uri insert(Uri uri, ContentValues values) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        long id = db.insert(DatabaseHelper.TABLE_USERS, null, values);
        
        if (id > 0) {
            Uri newUri = ContentUris.withAppendedId(CONTENT_URI, id);
            getContext().getContentResolver().notifyChange(newUri, null);
            return newUri;
        }
        
        throw new SQLException("Failed to insert row into " + uri);
    }
    
    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        int rowsUpdated;
        
        switch (uriMatcher.match(uri)) {
            case USERS:
                rowsUpdated = db.update(DatabaseHelper.TABLE_USERS, values, selection, selectionArgs);
                break;
            case USER_ID:
                String id = uri.getLastPathSegment();
                rowsUpdated = db.update(DatabaseHelper.TABLE_USERS, values,
                                      DatabaseHelper.COLUMN_ID + "=?", new String[]{id});
                break;
            default:
                throw new IllegalArgumentException("Unknown URI: " + uri);
        }
        
        if (rowsUpdated > 0) {
            getContext().getContentResolver().notifyChange(uri, null);
        }
        
        return rowsUpdated;
    }
    
    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        int rowsDeleted;
        
        switch (uriMatcher.match(uri)) {
            case USERS:
                rowsDeleted = db.delete(DatabaseHelper.TABLE_USERS, selection, selectionArgs);
                break;
            case USER_ID:
                String id = uri.getLastPathSegment();
                rowsDeleted = db.delete(DatabaseHelper.TABLE_USERS,
                                      DatabaseHelper.COLUMN_ID + "=?", new String[]{id});
                break;
            default:
                throw new IllegalArgumentException("Unknown URI: " + uri);
        }
        
        if (rowsDeleted > 0) {
            getContext().getContentResolver().notifyChange(uri, null);
        }
        
        return rowsDeleted;
    }
    
    @Override
    public String getType(Uri uri) {
        switch (uriMatcher.match(uri)) {
            case USERS:
                return "vnd.android.cursor.dir/vnd.example.users";
            case USER_ID:
                return "vnd.android.cursor.item/vnd.example.user";
            default:
                throw new IllegalArgumentException("Unknown URI: " + uri);
        }
    }
}
```

### Using Content Provider
```java
// Get ContentResolver
ContentResolver resolver = getContentResolver();

// Query
Cursor cursor = resolver.query(
    MyContentProvider.CONTENT_URI,
    null,
    null,
    null,
    null
);

// Insert
ContentValues values = new ContentValues();
values.put("name", "John Doe");
values.put("email", "john@example.com");
Uri newUri = resolver.insert(MyContentProvider.CONTENT_URI, values);

// Update
ContentValues updateValues = new ContentValues();
updateValues.put("age", 26);
int rowsUpdated = resolver.update(
    MyContentProvider.CONTENT_URI,
    updateValues,
    "name = ?",
    new String[]{"John Doe"}
);

// Delete
int rowsDeleted = resolver.delete(
    MyContentProvider.CONTENT_URI,
    "name = ?",
    new String[]{"John Doe"}
);
```

### Content Observer
```java
// Register content observer
ContentObserver observer = new ContentObserver(new Handler()) {
    @Override
    public void onChange(boolean selfChange) {
        // Handle data change
    }
};

getContentResolver().registerContentObserver(
    MyContentProvider.CONTENT_URI,
    true,
    observer
);

// Unregister observer
getContentResolver().unregisterContentObserver(observer);
```

## Network Storage

### HTTP Requests with HttpURLConnection
```java
private class NetworkTask extends AsyncTask<String, Void, String> {
    @Override
    protected String doInBackground(String... urls) {
        try {
            URL url = new URL(urls[0]);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(10000);
            
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader reader = new BufferedReader(
                    new InputStreamReader(connection.getInputStream())
                );
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    response.append(line);
                }
                reader.close();
                return response.toString();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
    
    @Override
    protected void onPostExecute(String result) {
        // Handle result on UI thread
    }
}

// Execute task
new NetworkTask().execute("https://api.example.com/data");
```

### POST Request
```java
try {
    URL url = new URL("https://api.example.com/users");
    HttpURLConnection connection = (HttpURLConnection) url.openConnection();
    connection.setRequestMethod("POST");
    connection.setRequestProperty("Content-Type", "application/json");
    connection.setDoOutput(true);
    
    // JSON data
    JSONObject jsonData = new JSONObject();
    jsonData.put("name", "John Doe");
    jsonData.put("email", "john@example.com");
    
    // Write data
    OutputStream os = connection.getOutputStream();
    os.write(jsonData.toString().getBytes());
    os.close();
    
    // Read response
    int responseCode = connection.getResponseCode();
    // Handle response...
    
} catch (IOException | JSONException e) {
    e.printStackTrace();
}
```

### Using Volley Library
```java
// Add to build.gradle
// implementation 'com.android.volley:volley:1.2.1'

// Create request queue
RequestQueue queue = Volley.newRequestQueue(this);

// String request
StringRequest stringRequest = new StringRequest(Request.Method.GET, url,
    new Response.Listener<String>() {
        @Override
        public void onResponse(String response) {
            // Handle successful response
        }
    },
    new Response.ErrorListener() {
        @Override
        public void onErrorResponse(VolleyError error) {
            // Handle error
        }
    }
);

// Add request to queue
queue.add(stringRequest);

// JSON Object request
JsonObjectRequest jsonRequest = new JsonObjectRequest(Request.Method.POST, url, jsonData,
    new Response.Listener<JSONObject>() {
        @Override
        public void onResponse(JSONObject response) {
            // Handle JSON response
        }
    },
    new Response.ErrorListener() {
        @Override
        public void onErrorResponse(VolleyError error) {
            // Handle error
        }
    }
);

queue.add(jsonRequest);
```

## Storage Permissions

### Manifest Permissions
```xml
<!-- External Storage (API < 19 or accessing other app directories) -->
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />

<!-- Internet access -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

<!-- Camera (if storing camera images) -->
<uses-permission android:name="android.permission.CAMERA" />

<!-- Legacy external storage (Android 10+) -->
<application android:requestLegacyExternalStorage="true">
```

### Runtime Permissions (API 23+)
```java
// Check permission
if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) 
    != PackageManager.PERMISSION_GRANTED) {
    
    // Request permission
    ActivityCompat.requestPermissions(this,
        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
        REQUEST_WRITE_STORAGE);
}

// Handle permission result
@Override
public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    
    if (requestCode == REQUEST_WRITE_STORAGE) {
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            // Permission granted
        } else {
            // Permission denied
        }
    }
}
```

## Best Practices

### Storage Selection Guidelines
```java
// Use SharedPreferences for:
// - User preferences and settings
// - Simple key-value pairs
// - Small amounts of data
// - Boolean flags and simple state

// Use Internal Storage for:
// - App-private files
// - Temporary files and cache
// - Files that shouldn't be accessible to other apps
// - Sensitive data

// Use External Storage for:
// - Large files (videos, images, documents)
// - Files to be shared with other apps
// - Files that should survive app uninstall
// - Media files

// Use SQLite/Room for:
// - Structured data with relationships
// - Data requiring complex queries
// - Large datasets that need indexing
// - Data that benefits from transactions

// Use Content Providers for:
// - Sharing data between apps
// - Providing controlled access to app data
// - Implementing standard data access patterns

// Use Network Storage for:
// - Data synchronization across devices
// - Backup and restore functionality
// - Cloud-based features
// - Real-time collaborative features
```

### Performance Tips
```java
// SharedPreferences
// - Use apply() instead of commit() when return value isn't needed
// - Avoid storing large strings or complex objects
// - Consider using multiple preference files for different categories

// File I/O
// - Use BufferedReader/BufferedWriter for better performance
// - Close streams in finally blocks or use try-with-resources
// - Avoid file operations on main thread

// Database
// - Use transactions for multiple operations
// - Create indexes for frequently queried columns
// - Use prepared statements for repeated queries
// - Close cursors and database connections

// Network
// - Implement proper error handling and retry logic
// - Use connection pooling libraries (OkHttp, Volley)
// - Cache responses when appropriate
// - Implement offline functionality
```

### Security Considerations
```java
// Internal Storage
// - Files are private by default
// - No additional security needed for non-sensitive data

// External Storage
// - Files are public and accessible by other apps
// - Don't store sensitive data
// - Validate file types and sizes

// SharedPreferences
// - Data is stored in plain text
// - Encrypt sensitive data before storing

// Network
// - Use HTTPS for all network communication
// - Validate server certificates
// - Implement proper authentication
// - Don't hardcode API keys in code

// Database
// - Use parameterized queries to prevent SQL injection
// - Consider encrypting sensitive database content
// - Implement proper access controls
```

### Common Patterns
```java
// Singleton pattern for database access
public class DatabaseManager {
    private static DatabaseManager instance;
    private DatabaseHelper dbHelper;
    
    private DatabaseManager(Context context) {
        dbHelper = new DatabaseHelper(context);
    }
    
    public static synchronized DatabaseManager getInstance(Context context) {
        if (instance == null) {
            instance = new DatabaseManager(context.getApplicationContext());
        }
        return instance;
    }
}

// Repository pattern
public class UserRepository {
    private UserDao userDao;
    private LiveData<List<User>> allUsers;
    
    public UserRepository(Application application) {
        AppDatabase db = AppDatabase.getDatabase(application);
        userDao = db.userDao();
        allUsers = userDao.getAllLive();
    }
    
    public LiveData<List<User>> getAllUsers() {
        return allUsers;
    }
    
    public void insert(User user) {
        AppDatabase.databaseWriteExecutor.execute(() -> {
            userDao.insert(user);
        });
    }
}

// Preference manager utility
public class PreferenceManager {
    private static final String PREF_NAME = "app_preferences";
    private static final String KEY_USER_ID = "user_id";
    private static final String KEY_IS_LOGGED_IN = "is_logged_in";
    
    private SharedPreferences prefs;
    
    public PreferenceManager(Context context) {
        prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }
    
    public void setUserId(int userId) {
        prefs.edit().putInt(KEY_USER_ID, userId).apply();
    }
    
    public int getUserId() {
        return prefs.getInt(KEY_USER_ID, -1);
    }
    
    public void setLoggedIn(boolean isLoggedIn) {
        prefs.edit().putBoolean(KEY_IS_LOGGED_IN, isLoggedIn).apply();
    }
    
    public boolean isLoggedIn() {
        return prefs.getBoolean(KEY_IS_LOGGED_IN, false);
    }
}
```

This comprehensive data storage cheat sheet covers all major storage options in Android development, perfect for written exams and quick reference during development.
