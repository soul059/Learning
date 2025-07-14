# Data Storage

## Table of Contents
- [Storage Options Overview](#storage-options-overview)
- [SharedPreferences](#sharedpreferences)
- [Internal Storage](#internal-storage)
- [External Storage](#external-storage)
- [SQLite Database](#sqlite-database)
- [Room Database](#room-database)
- [Content Providers](#content-providers)
- [Best Practices](#best-practices)

## Storage Options Overview

Android provides several options for storing data:

| Storage Type | Use Case | Accessibility | Persistence |
|--------------|----------|---------------|-------------|
| SharedPreferences | Small key-value pairs | Private to app | Survives app restart |
| Internal Storage | Private files | Private to app | Deleted when app uninstalled |
| External Storage | Large files, media | Can be public | May survive app uninstall |
| SQLite | Complex relational data | Private to app | Survives app restart |
| Room | Modern SQLite wrapper | Private to app | Survives app restart |
| Content Provider | Share data between apps | Can be public | Depends on implementation |

## SharedPreferences

**SharedPreferences** is perfect for storing small amounts of primitive data in key-value pairs.

### Basic Usage
```java
public class PreferencesManager {
    private static final String PREF_NAME = "MyAppPreferences";
    private static final String KEY_USER_NAME = "user_name";
    private static final String KEY_USER_ID = "user_id";
    private static final String KEY_IS_LOGGED_IN = "is_logged_in";
    private static final String KEY_THEME_MODE = "theme_mode";
    
    private SharedPreferences preferences;
    private SharedPreferences.Editor editor;
    
    public PreferencesManager(Context context) {
        preferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
        editor = preferences.edit();
    }
    
    // Save data
    public void saveUserInfo(String userName, int userId) {
        editor.putString(KEY_USER_NAME, userName);
        editor.putInt(KEY_USER_ID, userId);
        editor.putBoolean(KEY_IS_LOGGED_IN, true);
        editor.apply(); // or editor.commit() for synchronous
    }
    
    // Retrieve data
    public String getUserName() {
        return preferences.getString(KEY_USER_NAME, ""); // Default value
    }
    
    public int getUserId() {
        return preferences.getInt(KEY_USER_ID, -1);
    }
    
    public boolean isLoggedIn() {
        return preferences.getBoolean(KEY_IS_LOGGED_IN, false);
    }
    
    // Clear data
    public void logout() {
        editor.clear();
        editor.apply();
    }
    
    // Remove specific key
    public void removeUser() {
        editor.remove(KEY_USER_NAME);
        editor.remove(KEY_USER_ID);
        editor.apply();
    }
}
```

### Usage in Activity
```java
public class MainActivity extends AppCompatActivity {
    private PreferencesManager preferencesManager;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        preferencesManager = new PreferencesManager(this);
        
        // Check if user is logged in
        if (preferencesManager.isLoggedIn()) {
            String userName = preferencesManager.getUserName();
            welcomeUser(userName);
        } else {
            showLoginScreen();
        }
    }
    
    private void performLogin(String userName, int userId) {
        // Save login info
        preferencesManager.saveUserInfo(userName, userId);
        
        // Navigate to main screen
        startMainActivity();
    }
}
```

### Advanced SharedPreferences
```java
public class AdvancedPreferencesManager {
    
    // Store complex objects as JSON
    public void saveUserObject(User user) {
        Gson gson = new Gson();
        String json = gson.toJson(user);
        editor.putString("user_object", json);
        editor.apply();
    }
    
    public User getUserObject() {
        String json = preferences.getString("user_object", null);
        if (json != null) {
            Gson gson = new Gson();
            return gson.fromJson(json, User.class);
        }
        return null;
    }
    
    // Store arrays
    public void saveStringList(List<String> list) {
        Set<String> set = new HashSet<>(list);
        editor.putStringSet("string_list", set);
        editor.apply();
    }
    
    public List<String> getStringList() {
        Set<String> set = preferences.getStringSet("string_list", new HashSet<>());
        return new ArrayList<>(set);
    }
    
    // Encrypted SharedPreferences (API 23+)
    public static SharedPreferences getEncryptedPreferences(Context context) {
        try {
            MasterKeys.AliasSpec keyGenParameterSpec = MasterKeys.AES256_GCM_SPEC;
            String mainKeyAlias = MasterKeys.getOrCreate(keyGenParameterSpec);
            
            return EncryptedSharedPreferences.create(
                "encrypted_prefs",
                mainKeyAlias,
                context,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            );
        } catch (Exception e) {
            // Fallback to regular SharedPreferences
            return context.getSharedPreferences("regular_prefs", Context.MODE_PRIVATE);
        }
    }
}
```

## Internal Storage

**Internal Storage** is private to your app and stored in the device's internal memory.

### File Operations
```java
public class InternalStorageManager {
    private Context context;
    
    public InternalStorageManager(Context context) {
        this.context = context;
    }
    
    // Write to internal storage
    public boolean writeToFile(String fileName, String data) {
        try {
            FileOutputStream fos = context.openFileOutput(fileName, Context.MODE_PRIVATE);
            fos.write(data.getBytes());
            fos.close();
            return true;
        } catch (IOException e) {
            Log.e("InternalStorage", "Error writing file", e);
            return false;
        }
    }
    
    // Read from internal storage
    public String readFromFile(String fileName) {
        try {
            FileInputStream fis = context.openFileInput(fileName);
            int size = fis.available();
            byte[] buffer = new byte[size];
            fis.read(buffer);
            fis.close();
            return new String(buffer);
        } catch (IOException e) {
            Log.e("InternalStorage", "Error reading file", e);
            return null;
        }
    }
    
    // Write complex data
    public boolean saveObject(String fileName, Object object) {
        try {
            FileOutputStream fos = context.openFileOutput(fileName, Context.MODE_PRIVATE);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(object);
            oos.close();
            fos.close();
            return true;
        } catch (IOException e) {
            Log.e("InternalStorage", "Error saving object", e);
            return false;
        }
    }
    
    // Read complex data
    public Object loadObject(String fileName) {
        try {
            FileInputStream fis = context.openFileInput(fileName);
            ObjectInputStream ois = new ObjectInputStream(fis);
            Object object = ois.readObject();
            ois.close();
            fis.close();
            return object;
        } catch (IOException | ClassNotFoundException e) {
            Log.e("InternalStorage", "Error loading object", e);
            return null;
        }
    }
    
    // File management
    public boolean fileExists(String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        return file.exists();
    }
    
    public boolean deleteFile(String fileName) {
        return context.deleteFile(fileName);
    }
    
    public String[] getFileList() {
        return context.fileList();
    }
    
    // Work with subdirectories
    public File createSubDirectory(String dirName) {
        File dir = new File(context.getFilesDir(), dirName);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return dir;
    }
    
    // Cache directory (can be cleared by system)
    public boolean writeToCacheFile(String fileName, String data) {
        try {
            File cacheFile = new File(context.getCacheDir(), fileName);
            FileWriter writer = new FileWriter(cacheFile);
            writer.write(data);
            writer.close();
            return true;
        } catch (IOException e) {
            Log.e("InternalStorage", "Error writing cache file", e);
            return false;
        }
    }
}
```

### Usage Examples
```java
public class DataManager {
    private InternalStorageManager storageManager;
    
    public DataManager(Context context) {
        storageManager = new InternalStorageManager(context);
    }
    
    public void saveUserData(User user) {
        // Convert to JSON
        Gson gson = new Gson();
        String json = gson.toJson(user);
        
        // Save to file
        boolean success = storageManager.writeToFile("user_data.json", json);
        if (success) {
            Log.d("DataManager", "User data saved successfully");
        }
    }
    
    public User loadUserData() {
        String json = storageManager.readFromFile("user_data.json");
        if (json != null) {
            Gson gson = new Gson();
            return gson.fromJson(json, User.class);
        }
        return null;
    }
    
    public void saveAppLog(String logMessage) {
        String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", 
            Locale.getDefault()).format(new Date());
        String logEntry = timestamp + ": " + logMessage + "\n";
        
        // Append to log file
        try {
            FileOutputStream fos = context.openFileOutput("app_log.txt", 
                Context.MODE_APPEND);
            fos.write(logEntry.getBytes());
            fos.close();
        } catch (IOException e) {
            Log.e("DataManager", "Error writing log", e);
        }
    }
}
```

## External Storage

**External Storage** includes SD cards and other external storage media.

### Permission Requirements
```xml
<!-- AndroidManifest.xml -->
<!-- For reading external storage -->
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />

<!-- For writing external storage (API level < 19) -->
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" 
    android:maxSdkVersion="18" />

<!-- For managing external storage (API level 30+) -->
<uses-permission android:name="android.permission.MANAGE_EXTERNAL_STORAGE"
    tools:ignore="ScopedStorage" />
```

### External Storage Manager
```java
public class ExternalStorageManager {
    private Context context;
    
    public ExternalStorageManager(Context context) {
        this.context = context;
    }
    
    // Check if external storage is available
    public boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        return Environment.MEDIA_MOUNTED.equals(state);
    }
    
    public boolean isExternalStorageReadable() {
        String state = Environment.getExternalStorageState();
        return Environment.MEDIA_MOUNTED.equals(state) ||
               Environment.MEDIA_MOUNTED_READ_ONLY.equals(state);
    }
    
    // Get external storage directories
    public File getPublicDocumentsDir() {
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);
    }
    
    public File getPublicPicturesDir() {
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
    }
    
    public File getPublicDownloadsDir() {
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
    }
    
    // Get app-specific external storage (no permission required for API 19+)
    public File getAppExternalFilesDir() {
        return context.getExternalFilesDir(null);
    }
    
    public File getAppExternalPicturesDir() {
        return context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
    }
    
    public File getAppExternalCacheDir() {
        return context.getExternalCacheDir();
    }
    
    // Write to external storage
    public boolean writeToExternalFile(String fileName, String data) {
        if (!isExternalStorageWritable()) {
            return false;
        }
        
        try {
            File file = new File(getAppExternalFilesDir(), fileName);
            FileWriter writer = new FileWriter(file);
            writer.write(data);
            writer.close();
            return true;
        } catch (IOException e) {
            Log.e("ExternalStorage", "Error writing file", e);
            return false;
        }
    }
    
    // Save bitmap to external storage
    public boolean saveBitmapToExternalStorage(Bitmap bitmap, String fileName) {
        if (!isExternalStorageWritable()) {
            return false;
        }
        
        try {
            File picturesDir = getAppExternalPicturesDir();
            if (!picturesDir.exists()) {
                picturesDir.mkdirs();
            }
            
            File file = new File(picturesDir, fileName);
            FileOutputStream fos = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.close();
            
            // Add to media store
            MediaScannerConnection.scanFile(context, new String[]{file.getAbsolutePath()}, 
                null, null);
            
            return true;
        } catch (IOException e) {
            Log.e("ExternalStorage", "Error saving bitmap", e);
            return false;
        }
    }
    
    // Modern approach for Android 10+ (Scoped Storage)
    public Uri saveImageToMediaStore(Bitmap bitmap, String displayName) {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, displayName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES);
        
        ContentResolver resolver = context.getContentResolver();
        Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        
        if (uri != null) {
            try {
                OutputStream outputStream = resolver.openOutputStream(uri);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
                outputStream.close();
                return uri;
            } catch (IOException e) {
                Log.e("ExternalStorage", "Error saving to media store", e);
                resolver.delete(uri, null, null);
            }
        }
        
        return null;
    }
}
```

## SQLite Database

**SQLite** is a lightweight database engine perfect for local data storage.

### Database Helper
```java
public class DatabaseHelper extends SQLiteOpenHelper {
    
    private static final String DATABASE_NAME = "MyApp.db";
    private static final int DATABASE_VERSION = 1;
    
    // Table names
    public static final String TABLE_USERS = "users";
    public static final String TABLE_PRODUCTS = "products";
    
    // User table columns
    public static final String COLUMN_USER_ID = "id";
    public static final String COLUMN_USER_NAME = "name";
    public static final String COLUMN_USER_EMAIL = "email";
    public static final String COLUMN_USER_PHONE = "phone";
    public static final String COLUMN_USER_CREATED_AT = "created_at";
    
    // Product table columns
    public static final String COLUMN_PRODUCT_ID = "id";
    public static final String COLUMN_PRODUCT_NAME = "name";
    public static final String COLUMN_PRODUCT_PRICE = "price";
    public static final String COLUMN_PRODUCT_CATEGORY = "category";
    
    // Create table statements
    private static final String CREATE_TABLE_USERS = 
        "CREATE TABLE " + TABLE_USERS + "(" +
        COLUMN_USER_ID + " INTEGER PRIMARY KEY AUTOINCREMENT," +
        COLUMN_USER_NAME + " TEXT NOT NULL," +
        COLUMN_USER_EMAIL + " TEXT UNIQUE," +
        COLUMN_USER_PHONE + " TEXT," +
        COLUMN_USER_CREATED_AT + " DATETIME DEFAULT CURRENT_TIMESTAMP" +
        ")";
    
    private static final String CREATE_TABLE_PRODUCTS = 
        "CREATE TABLE " + TABLE_PRODUCTS + "(" +
        COLUMN_PRODUCT_ID + " INTEGER PRIMARY KEY AUTOINCREMENT," +
        COLUMN_PRODUCT_NAME + " TEXT NOT NULL," +
        COLUMN_PRODUCT_PRICE + " REAL," +
        COLUMN_PRODUCT_CATEGORY + " TEXT" +
        ")";
    
    public DatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }
    
    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(CREATE_TABLE_USERS);
        db.execSQL(CREATE_TABLE_PRODUCTS);
    }
    
    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // Handle database upgrades
        if (oldVersion < 2) {
            // Add new column or table for version 2
            db.execSQL("ALTER TABLE " + TABLE_USERS + " ADD COLUMN avatar_url TEXT");
        }
        
        // For major upgrades, you might want to recreate tables
        // db.execSQL("DROP TABLE IF EXISTS " + TABLE_USERS);
        // onCreate(db);
    }
}
```

### DAO (Data Access Object)
```java
public class UserDAO {
    private DatabaseHelper dbHelper;
    private SQLiteDatabase database;
    
    public UserDAO(Context context) {
        dbHelper = new DatabaseHelper(context);
    }
    
    public void open() throws SQLException {
        database = dbHelper.getWritableDatabase();
    }
    
    public void close() {
        dbHelper.close();
    }
    
    // Create
    public long insertUser(User user) {
        ContentValues values = new ContentValues();
        values.put(DatabaseHelper.COLUMN_USER_NAME, user.getName());
        values.put(DatabaseHelper.COLUMN_USER_EMAIL, user.getEmail());
        values.put(DatabaseHelper.COLUMN_USER_PHONE, user.getPhone());
        
        return database.insert(DatabaseHelper.TABLE_USERS, null, values);
    }
    
    // Read
    public User getUserById(long id) {
        Cursor cursor = database.query(
            DatabaseHelper.TABLE_USERS,
            null, // all columns
            DatabaseHelper.COLUMN_USER_ID + " = ?",
            new String[]{String.valueOf(id)},
            null, null, null
        );
        
        User user = null;
        if (cursor.moveToFirst()) {
            user = cursorToUser(cursor);
        }
        cursor.close();
        return user;
    }
    
    public List<User> getAllUsers() {
        List<User> users = new ArrayList<>();
        
        Cursor cursor = database.query(
            DatabaseHelper.TABLE_USERS,
            null, null, null, null, null,
            DatabaseHelper.COLUMN_USER_NAME + " ASC"
        );
        
        cursor.moveToFirst();
        while (!cursor.isAfterLast()) {
            User user = cursorToUser(cursor);
            users.add(user);
            cursor.moveToNext();
        }
        cursor.close();
        return users;
    }
    
    // Update
    public int updateUser(User user) {
        ContentValues values = new ContentValues();
        values.put(DatabaseHelper.COLUMN_USER_NAME, user.getName());
        values.put(DatabaseHelper.COLUMN_USER_EMAIL, user.getEmail());
        values.put(DatabaseHelper.COLUMN_USER_PHONE, user.getPhone());
        
        return database.update(
            DatabaseHelper.TABLE_USERS,
            values,
            DatabaseHelper.COLUMN_USER_ID + " = ?",
            new String[]{String.valueOf(user.getId())}
        );
    }
    
    // Delete
    public void deleteUser(long id) {
        database.delete(
            DatabaseHelper.TABLE_USERS,
            DatabaseHelper.COLUMN_USER_ID + " = ?",
            new String[]{String.valueOf(id)}
        );
    }
    
    // Search
    public List<User> searchUsers(String searchTerm) {
        List<User> users = new ArrayList<>();
        
        String selection = DatabaseHelper.COLUMN_USER_NAME + " LIKE ? OR " +
                          DatabaseHelper.COLUMN_USER_EMAIL + " LIKE ?";
        String[] selectionArgs = {"%" + searchTerm + "%", "%" + searchTerm + "%"};
        
        Cursor cursor = database.query(
            DatabaseHelper.TABLE_USERS,
            null, selection, selectionArgs,
            null, null, DatabaseHelper.COLUMN_USER_NAME + " ASC"
        );
        
        cursor.moveToFirst();
        while (!cursor.isAfterLast()) {
            User user = cursorToUser(cursor);
            users.add(user);
            cursor.moveToNext();
        }
        cursor.close();
        return users;
    }
    
    // Helper method to convert cursor to User object
    private User cursorToUser(Cursor cursor) {
        User user = new User();
        user.setId(cursor.getLong(cursor.getColumnIndex(DatabaseHelper.COLUMN_USER_ID)));
        user.setName(cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_USER_NAME)));
        user.setEmail(cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_USER_EMAIL)));
        user.setPhone(cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_USER_PHONE)));
        
        String createdAt = cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_USER_CREATED_AT));
        user.setCreatedAt(createdAt);
        
        return user;
    }
    
    // Raw SQL queries
    public List<User> getUsersWithCustomQuery() {
        List<User> users = new ArrayList<>();
        
        String query = "SELECT * FROM " + DatabaseHelper.TABLE_USERS + 
                      " WHERE " + DatabaseHelper.COLUMN_USER_CREATED_AT + 
                      " > datetime('now', '-30 days') ORDER BY " + 
                      DatabaseHelper.COLUMN_USER_NAME;
        
        Cursor cursor = database.rawQuery(query, null);
        
        cursor.moveToFirst();
        while (!cursor.isAfterLast()) {
            User user = cursorToUser(cursor);
            users.add(user);
            cursor.moveToNext();
        }
        cursor.close();
        return users;
    }
}
```

### Usage in Activity
```java
public class MainActivity extends AppCompatActivity {
    private UserDAO userDAO;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        userDAO = new UserDAO(this);
        userDAO.open();
        
        // Example operations
        insertSampleData();
        displayUsers();
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (userDAO != null) {
            userDAO.close();
        }
    }
    
    private void insertSampleData() {
        User user1 = new User("John Doe", "john@example.com", "123-456-7890");
        User user2 = new User("Jane Smith", "jane@example.com", "098-765-4321");
        
        long id1 = userDAO.insertUser(user1);
        long id2 = userDAO.insertUser(user2);
        
        Log.d("Database", "Inserted users with IDs: " + id1 + ", " + id2);
    }
    
    private void displayUsers() {
        List<User> users = userDAO.getAllUsers();
        for (User user : users) {
            Log.d("Database", "User: " + user.getName() + " - " + user.getEmail());
        }
    }
}
```

## Room Database

**Room** is the recommended way to work with SQLite in modern Android development.

### Dependencies
```gradle
// app/build.gradle
dependencies {
    implementation "androidx.room:room-runtime:2.5.0"
    annotationProcessor "androidx.room:room-compiler:2.5.0"
    
    // Optional - RxJava support
    implementation "androidx.room:room-rxjava2:2.5.0"
    
    // Optional - Guava support
    implementation "androidx.room:room-guava:2.5.0"
    
    // Test helpers
    testImplementation "androidx.room:room-testing:2.5.0"
}
```

### Entity
```java
@Entity(tableName = "users")
public class User {
    @PrimaryKey(autoGenerate = true)
    private int id;
    
    @ColumnInfo(name = "name")
    private String name;
    
    @ColumnInfo(name = "email")
    private String email;
    
    @ColumnInfo(name = "phone")
    private String phone;
    
    @ColumnInfo(name = "created_at")
    private Date createdAt;
    
    // Constructors
    public User() {
        this.createdAt = new Date();
    }
    
    public User(String name, String email, String phone) {
        this.name = name;
        this.email = email;
        this.phone = phone;
        this.createdAt = new Date();
    }
    
    // Getters and setters
    public int getId() { return id; }
    public void setId(int id) { this.id = id; }
    
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    
    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }
    
    public Date getCreatedAt() { return createdAt; }
    public void setCreatedAt(Date createdAt) { this.createdAt = createdAt; }
}

// Type converters for complex data types
@Entity
public class UserWithConverters {
    @PrimaryKey(autoGenerate = true)
    private int id;
    
    private String name;
    
    @TypeConverters(DateConverter.class)
    private Date createdAt;
    
    @TypeConverters(StringListConverter.class)
    private List<String> tags;
    
    // Constructors, getters, setters...
}

// Date converter
public class DateConverter {
    @TypeConverter
    public static Date fromTimestamp(Long value) {
        return value == null ? null : new Date(value);
    }
    
    @TypeConverter
    public static Long dateToTimestamp(Date date) {
        return date == null ? null : date.getTime();
    }
}

// String list converter
public class StringListConverter {
    @TypeConverter
    public static List<String> fromString(String value) {
        Type listType = new TypeToken<List<String>>(){}.getType();
        return new Gson().fromJson(value, listType);
    }
    
    @TypeConverter
    public static String fromStringList(List<String> list) {
        return new Gson().toJson(list);
    }
}
```

### DAO (Data Access Object)
```java
@Dao
public interface UserDao {
    
    // Insert
    @Insert
    long insert(User user);
    
    @Insert
    void insertAll(User... users);
    
    @Insert
    List<Long> insertAll(List<User> users);
    
    // Update
    @Update
    void update(User user);
    
    // Delete
    @Delete
    void delete(User user);
    
    @Query("DELETE FROM users WHERE id = :id")
    void deleteById(int id);
    
    @Query("DELETE FROM users")
    void deleteAll();
    
    // Select queries
    @Query("SELECT * FROM users")
    List<User> getAllUsers();
    
    @Query("SELECT * FROM users WHERE id = :id")
    User getUserById(int id);
    
    @Query("SELECT * FROM users WHERE name LIKE :name")
    List<User> getUsersByName(String name);
    
    @Query("SELECT * FROM users WHERE email = :email")
    User getUserByEmail(String email);
    
    @Query("SELECT * FROM users ORDER BY name ASC")
    List<User> getUsersOrderedByName();
    
    @Query("SELECT * FROM users WHERE created_at > :date")
    List<User> getUsersCreatedAfter(Date date);
    
    // Complex queries
    @Query("SELECT * FROM users WHERE name LIKE '%' || :searchTerm || '%' OR email LIKE '%' || :searchTerm || '%'")
    List<User> searchUsers(String searchTerm);
    
    @Query("SELECT COUNT(*) FROM users")
    int getUserCount();
    
    @Query("SELECT * FROM users LIMIT :limit OFFSET :offset")
    List<User> getUsersWithPagination(int limit, int offset);
    
    // LiveData for observing changes
    @Query("SELECT * FROM users")
    LiveData<List<User>> getAllUsersLiveData();
    
    @Query("SELECT * FROM users WHERE id = :id")
    LiveData<User> getUserByIdLiveData(int id);
}
```

### Database Class
```java
@Database(
    entities = {User.class, Product.class},
    version = 1,
    exportSchema = false
)
@TypeConverters({DateConverter.class, StringListConverter.class})
public abstract class AppDatabase extends RoomDatabase {
    
    private static volatile AppDatabase INSTANCE;
    
    public abstract UserDao userDao();
    public abstract ProductDao productDao();
    
    public static AppDatabase getDatabase(final Context context) {
        if (INSTANCE == null) {
            synchronized (AppDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(
                        context.getApplicationContext(),
                        AppDatabase.class,
                        "app_database"
                    )
                    .addCallback(roomDatabaseCallback)
                    .build();
                }
            }
        }
        return INSTANCE;
    }
    
    // Callback for database creation
    private static RoomDatabase.Callback roomDatabaseCallback = new RoomDatabase.Callback() {
        @Override
        public void onCreate(@NonNull SupportSQLiteDatabase db) {
            super.onCreate(db);
            // Populate database with initial data
            new PopulateDbAsync(INSTANCE).execute();
        }
    };
    
    // AsyncTask to populate database
    private static class PopulateDbAsync extends AsyncTask<Void, Void, Void> {
        private final UserDao userDao;
        
        PopulateDbAsync(AppDatabase db) {
            userDao = db.userDao();
        }
        
        @Override
        protected Void doInBackground(final Void... params) {
            userDao.deleteAll();
            
            User user1 = new User("John Doe", "john@example.com", "123-456-7890");
            User user2 = new User("Jane Smith", "jane@example.com", "098-765-4321");
            
            userDao.insert(user1);
            userDao.insert(user2);
            
            return null;
        }
    }
}
```

### Repository Pattern
```java
public class UserRepository {
    private UserDao userDao;
    private LiveData<List<User>> allUsers;
    
    public UserRepository(Application application) {
        AppDatabase db = AppDatabase.getDatabase(application);
        userDao = db.userDao();
        allUsers = userDao.getAllUsersLiveData();
    }
    
    // Observed LiveData
    public LiveData<List<User>> getAllUsers() {
        return allUsers;
    }
    
    public LiveData<User> getUserById(int id) {
        return userDao.getUserByIdLiveData(id);
    }
    
    // Database operations must be done in background thread
    public void insert(User user) {
        new InsertAsyncTask(userDao).execute(user);
    }
    
    public void update(User user) {
        new UpdateAsyncTask(userDao).execute(user);
    }
    
    public void delete(User user) {
        new DeleteAsyncTask(userDao).execute(user);
    }
    
    // AsyncTask classes
    private static class InsertAsyncTask extends AsyncTask<User, Void, Void> {
        private UserDao asyncTaskDao;
        
        InsertAsyncTask(UserDao dao) {
            asyncTaskDao = dao;
        }
        
        @Override
        protected Void doInBackground(final User... params) {
            asyncTaskDao.insert(params[0]);
            return null;
        }
    }
    
    private static class UpdateAsyncTask extends AsyncTask<User, Void, Void> {
        private UserDao asyncTaskDao;
        
        UpdateAsyncTask(UserDao dao) {
            asyncTaskDao = dao;
        }
        
        @Override
        protected Void doInBackground(final User... params) {
            asyncTaskDao.update(params[0]);
            return null;
        }
    }
    
    private static class DeleteAsyncTask extends AsyncTask<User, Void, Void> {
        private UserDao asyncTaskDao;
        
        DeleteAsyncTask(UserDao dao) {
            asyncTaskDao = dao;
        }
        
        @Override
        protected Void doInBackground(final User... params) {
            asyncTaskDao.delete(params[0]);
            return null;
        }
    }
}
```

### ViewModel with Room
```java
public class UserViewModel extends AndroidViewModel {
    
    private UserRepository repository;
    private LiveData<List<User>> allUsers;
    
    public UserViewModel(@NonNull Application application) {
        super(application);
        repository = new UserRepository(application);
        allUsers = repository.getAllUsers();
    }
    
    public LiveData<List<User>> getAllUsers() {
        return allUsers;
    }
    
    public LiveData<User> getUserById(int id) {
        return repository.getUserById(id);
    }
    
    public void insert(User user) {
        repository.insert(user);
    }
    
    public void update(User user) {
        repository.update(user);
    }
    
    public void delete(User user) {
        repository.delete(user);
    }
}
```

### Usage in Activity/Fragment
```java
public class MainActivity extends AppCompatActivity {
    
    private UserViewModel userViewModel;
    private RecyclerView recyclerView;
    private UserAdapter adapter;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize RecyclerView
        recyclerView = findViewById(R.id.recyclerView);
        adapter = new UserAdapter();
        recyclerView.setAdapter(adapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        
        // Initialize ViewModel
        userViewModel = new ViewModelProvider(this).get(UserViewModel.class);
        
        // Observe data changes
        userViewModel.getAllUsers().observe(this, users -> {
            // Update UI
            adapter.setUsers(users);
        });
        
        // Add new user button
        findViewById(R.id.buttonAddUser).setOnClickListener(v -> {
            User newUser = new User("New User", "new@example.com", "111-222-3333");
            userViewModel.insert(newUser);
        });
    }
}
```

## Content Providers

**Content Providers** manage access to a central repository of data and share data between applications.

### Creating a Content Provider
```java
public class UserContentProvider extends ContentProvider {
    
    private static final String AUTHORITY = "com.example.myapp.provider";
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
    public Cursor query(@NonNull Uri uri, @Nullable String[] projection, 
                       @Nullable String selection, @Nullable String[] selectionArgs, 
                       @Nullable String sortOrder) {
        
        SQLiteDatabase db = dbHelper.getReadableDatabase();
        Cursor cursor;
        
        switch (uriMatcher.match(uri)) {
            case USERS:
                cursor = db.query(
                    DatabaseHelper.TABLE_USERS,
                    projection, selection, selectionArgs,
                    null, null, sortOrder
                );
                break;
            case USER_ID:
                String id = uri.getLastPathSegment();
                cursor = db.query(
                    DatabaseHelper.TABLE_USERS,
                    projection,
                    DatabaseHelper.COLUMN_USER_ID + " = ?",
                    new String[]{id},
                    null, null, sortOrder
                );
                break;
            default:
                throw new IllegalArgumentException("Unknown URI: " + uri);
        }
        
        cursor.setNotificationUri(getContext().getContentResolver(), uri);
        return cursor;
    }
    
    @Override
    public Uri insert(@NonNull Uri uri, @Nullable ContentValues values) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        
        switch (uriMatcher.match(uri)) {
            case USERS:
                long id = db.insert(DatabaseHelper.TABLE_USERS, null, values);
                if (id > 0) {
                    Uri newUri = ContentUris.withAppendedId(CONTENT_URI, id);
                    getContext().getContentResolver().notifyChange(newUri, null);
                    return newUri;
                }
                break;
            default:
                throw new IllegalArgumentException("Unknown URI: " + uri);
        }
        
        return null;
    }
    
    @Override
    public int update(@NonNull Uri uri, @Nullable ContentValues values, 
                     @Nullable String selection, @Nullable String[] selectionArgs) {
        
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        int rowsUpdated;
        
        switch (uriMatcher.match(uri)) {
            case USERS:
                rowsUpdated = db.update(DatabaseHelper.TABLE_USERS, values, selection, selectionArgs);
                break;
            case USER_ID:
                String id = uri.getLastPathSegment();
                rowsUpdated = db.update(
                    DatabaseHelper.TABLE_USERS, values,
                    DatabaseHelper.COLUMN_USER_ID + " = ?",
                    new String[]{id}
                );
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
    public int delete(@NonNull Uri uri, @Nullable String selection, 
                     @Nullable String[] selectionArgs) {
        
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        int rowsDeleted;
        
        switch (uriMatcher.match(uri)) {
            case USERS:
                rowsDeleted = db.delete(DatabaseHelper.TABLE_USERS, selection, selectionArgs);
                break;
            case USER_ID:
                String id = uri.getLastPathSegment();
                rowsDeleted = db.delete(
                    DatabaseHelper.TABLE_USERS,
                    DatabaseHelper.COLUMN_USER_ID + " = ?",
                    new String[]{id}
                );
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
    public String getType(@NonNull Uri uri) {
        switch (uriMatcher.match(uri)) {
            case USERS:
                return "vnd.android.cursor.dir/vnd.com.example.myapp.users";
            case USER_ID:
                return "vnd.android.cursor.item/vnd.com.example.myapp.users";
            default:
                throw new IllegalArgumentException("Unknown URI: " + uri);
        }
    }
}
```

### Register Content Provider
```xml
<!-- AndroidManifest.xml -->
<provider
    android:name=".UserContentProvider"
    android:authorities="com.example.myapp.provider"
    android:exported="true"
    android:readPermission="com.example.myapp.READ_USERS"
    android:writePermission="com.example.myapp.WRITE_USERS" />

<!-- Define custom permissions -->
<permission
    android:name="com.example.myapp.READ_USERS"
    android:protectionLevel="normal" />

<permission
    android:name="com.example.myapp.WRITE_USERS"
    android:protectionLevel="normal" />
```

### Using Content Provider
```java
public class ContentProviderClient {
    
    private ContentResolver contentResolver;
    
    public ContentProviderClient(Context context) {
        contentResolver = context.getContentResolver();
    }
    
    // Query data
    public List<User> getAllUsers() {
        List<User> users = new ArrayList<>();
        
        Cursor cursor = contentResolver.query(
            UserContentProvider.CONTENT_URI,
            null, null, null,
            DatabaseHelper.COLUMN_USER_NAME + " ASC"
        );
        
        if (cursor != null) {
            while (cursor.moveToNext()) {
                User user = new User();
                user.setId(cursor.getInt(cursor.getColumnIndex(DatabaseHelper.COLUMN_USER_ID)));
                user.setName(cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_USER_NAME)));
                user.setEmail(cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_USER_EMAIL)));
                users.add(user);
            }
            cursor.close();
        }
        
        return users;
    }
    
    // Insert data
    public Uri insertUser(User user) {
        ContentValues values = new ContentValues();
        values.put(DatabaseHelper.COLUMN_USER_NAME, user.getName());
        values.put(DatabaseHelper.COLUMN_USER_EMAIL, user.getEmail());
        values.put(DatabaseHelper.COLUMN_USER_PHONE, user.getPhone());
        
        return contentResolver.insert(UserContentProvider.CONTENT_URI, values);
    }
    
    // Update data
    public int updateUser(User user) {
        ContentValues values = new ContentValues();
        values.put(DatabaseHelper.COLUMN_USER_NAME, user.getName());
        values.put(DatabaseHelper.COLUMN_USER_EMAIL, user.getEmail());
        
        Uri uri = ContentUris.withAppendedId(UserContentProvider.CONTENT_URI, user.getId());
        return contentResolver.update(uri, values, null, null);
    }
    
    // Delete data
    public int deleteUser(int userId) {
        Uri uri = ContentUris.withAppendedId(UserContentProvider.CONTENT_URI, userId);
        return contentResolver.delete(uri, null, null);
    }
}
```

## Best Practices

### 1. Choose Appropriate Storage
```java
// Use SharedPreferences for small key-value data
if (dataSize < 1KB && isKeyValuePair) {
    useSharedPreferences();
}

// Use Room for structured relational data
if (needsComplexQueries || needsRelationships) {
    useRoomDatabase();
}

// Use files for large unstructured data
if (isLargeFile || isMediaFile) {
    useFileStorage();
}
```

### 2. Handle Threading Properly
```java
// Never do database operations on UI thread
// Room automatically handles this with LiveData
userViewModel.getAllUsers().observe(this, users -> {
    // This runs on UI thread
    updateUI(users);
});

// For direct database access, use background threads
new AsyncTask<Void, Void, List<User>>() {
    @Override
    protected List<User> doInBackground(Void... voids) {
        return userDao.getAllUsers();
    }
    
    @Override
    protected void onPostExecute(List<User> users) {
        updateUI(users);
    }
}.execute();
```

### 3. Handle Errors Gracefully
```java
public boolean saveUserData(User user) {
    try {
        long id = userDao.insert(user);
        return id > 0;
    } catch (SQLiteConstraintException e) {
        Log.e("Database", "Constraint violation: " + e.getMessage());
        // Handle unique constraint violations
        return false;
    } catch (SQLException e) {
        Log.e("Database", "Database error: " + e.getMessage());
        return false;
    }
}
```

### 4. Optimize Performance
```java
// Use transactions for bulk operations
@Transaction
@Query("SELECT * FROM users WHERE department = :dept")
public List<UserWithProjects> getUsersWithProjects(String dept);

// Use indexes for frequently queried columns
@Entity(indices = {@Index("email"), @Index("name")})
public class User {
    // ...
}

// Limit query results when appropriate
@Query("SELECT * FROM users ORDER BY created_at DESC LIMIT 100")
List<User> getRecentUsers();
```

### 5. Security Considerations
```java
// Use encrypted storage for sensitive data
public void saveSensitiveData(String data) {
    try {
        SharedPreferences encryptedPrefs = EncryptedSharedPreferences.create(
            "secret_shared_prefs",
            masterKeyAlias,
            context,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        );
        
        encryptedPrefs.edit().putString("sensitive_key", data).apply();
    } catch (Exception e) {
        Log.e("Security", "Error saving encrypted data", e);
    }
}

// Validate data before storage
public boolean saveUser(User user) {
    if (user == null || TextUtils.isEmpty(user.getEmail())) {
        return false;
    }
    
    if (!android.util.Patterns.EMAIL_ADDRESS.matcher(user.getEmail()).matches()) {
        return false;
    }
    
    return userDao.insert(user) > 0;
}
```

Understanding different storage options and choosing the right one for your use case is crucial for building efficient Android applications. Each storage method has its own strengths and is suitable for different types of data and use cases.
