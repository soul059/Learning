# Networking and APIs

## Table of Contents
- [HTTP Networking Basics](#http-networking-basics)
- [HttpURLConnection](#httpurlconnection)
- [OkHttp Library](#okhttp-library)
- [Retrofit Library](#retrofit-library)
- [JSON Parsing](#json-parsing)
- [Image Loading](#image-loading)
- [Network Security](#network-security)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## HTTP Networking Basics

### Network Permissions
```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

<!-- For HTTP traffic in production (Android 9+) -->
<application
    android:usesCleartextTraffic="true"
    ... >
</application>

<!-- Or use network security config -->
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ... >
</application>
```

### Network Security Config
```xml
<!-- res/xml/network_security_config.xml -->
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">api.example.com</domain>
        <domain includeSubdomains="true">localhost</domain>
    </domain-config>
</network-security-config>
```

### Check Network Connection
```java
public class NetworkUtils {
    
    public static boolean isNetworkAvailable(Context context) {
        ConnectivityManager connectivityManager = 
            (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        
        if (connectivityManager != null) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                Network network = connectivityManager.getActiveNetwork();
                if (network != null) {
                    NetworkCapabilities networkCapabilities = 
                        connectivityManager.getNetworkCapabilities(network);
                    return networkCapabilities != null && 
                           (networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) || 
                            networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR));
                }
            } else {
                NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
                return activeNetworkInfo != null && activeNetworkInfo.isConnected();
            }
        }
        return false;
    }
    
    public static boolean isWifiConnected(Context context) {
        ConnectivityManager connectivityManager = 
            (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        
        if (connectivityManager != null) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                Network network = connectivityManager.getActiveNetwork();
                if (network != null) {
                    NetworkCapabilities networkCapabilities = 
                        connectivityManager.getNetworkCapabilities(network);
                    return networkCapabilities != null && 
                           networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI);
                }
            } else {
                NetworkInfo wifiInfo = connectivityManager.getNetworkInfo(ConnectivityManager.TYPE_WIFI);
                return wifiInfo != null && wifiInfo.isConnected();
            }
        }
        return false;
    }
}
```

## HttpURLConnection

### Basic GET Request
```java
public class HttpURLConnectionExample {
    
    public static String performGetRequest(String urlString) throws IOException {
        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        
        try {
            // Set request method
            connection.setRequestMethod("GET");
            
            // Set headers
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("User-Agent", "MyApp/1.0");
            
            // Set timeouts
            connection.setConnectTimeout(10000); // 10 seconds
            connection.setReadTimeout(15000);    // 15 seconds
            
            // Get response code
            int responseCode = connection.getResponseCode();
            
            if (responseCode == HttpURLConnection.HTTP_OK) {
                return readInputStream(connection.getInputStream());
            } else {
                throw new IOException("HTTP Error: " + responseCode);
            }
            
        } finally {
            connection.disconnect();
        }
    }
    
    public static String performPostRequest(String urlString, String jsonBody) throws IOException {
        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        
        try {
            // Set request method
            connection.setRequestMethod("POST");
            
            // Set headers
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Accept", "application/json");
            
            // Enable output for POST data
            connection.setDoOutput(true);
            
            // Write request body
            try (OutputStream outputStream = connection.getOutputStream()) {
                byte[] input = jsonBody.getBytes("utf-8");
                outputStream.write(input, 0, input.length);
            }
            
            // Get response
            int responseCode = connection.getResponseCode();
            
            if (responseCode == HttpURLConnection.HTTP_OK || 
                responseCode == HttpURLConnection.HTTP_CREATED) {
                return readInputStream(connection.getInputStream());
            } else {
                throw new IOException("HTTP Error: " + responseCode);
            }
            
        } finally {
            connection.disconnect();
        }
    }
    
    private static String readInputStream(InputStream inputStream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        StringBuilder result = new StringBuilder();
        String line;
        
        while ((line = reader.readLine()) != null) {
            result.append(line);
        }
        
        return result.toString();
    }
}
```

### AsyncTask for HTTP Requests
```java
public class HttpAsyncTask extends AsyncTask<String, Void, String> {
    
    private Exception exception;
    private OnResponseListener listener;
    
    public interface OnResponseListener {
        void onSuccess(String response);
        void onError(String error);
    }
    
    public HttpAsyncTask(OnResponseListener listener) {
        this.listener = listener;
    }
    
    @Override
    protected String doInBackground(String... urls) {
        try {
            return HttpURLConnectionExample.performGetRequest(urls[0]);
        } catch (Exception e) {
            this.exception = e;
            return null;
        }
    }
    
    @Override
    protected void onPostExecute(String result) {
        if (exception == null && listener != null) {
            listener.onSuccess(result);
        } else if (listener != null) {
            listener.onError(exception.getMessage());
        }
    }
}

// Usage
new HttpAsyncTask(new HttpAsyncTask.OnResponseListener() {
    @Override
    public void onSuccess(String response) {
        // Handle successful response
        Log.d("HTTP", "Response: " + response);
    }
    
    @Override
    public void onError(String error) {
        // Handle error
        Log.e("HTTP", "Error: " + error);
    }
}).execute("https://api.example.com/users");
```

## OkHttp Library

### Add Dependency
```gradle
// app/build.gradle
dependencies {
    implementation 'com.squareup.okhttp3:okhttp:4.11.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.11.0'
}
```

### Basic OkHttp Usage
```java
public class OkHttpManager {
    
    private OkHttpClient client;
    
    public OkHttpManager() {
        // Create HTTP logging interceptor
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);
        
        // Build client with interceptors and timeouts
        client = new OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();
    }
    
    // GET request
    public void performGet(String url, Callback callback) {
        Request request = new Request.Builder()
            .url(url)
            .addHeader("Authorization", "Bearer " + getAuthToken())
            .addHeader("User-Agent", "MyApp/1.0")
            .build();
        
        client.newCall(request).enqueue(callback);
    }
    
    // POST request with JSON
    public void performPost(String url, String json, Callback callback) {
        RequestBody body = RequestBody.create(json, 
            MediaType.get("application/json; charset=utf-8"));
        
        Request request = new Request.Builder()
            .url(url)
            .post(body)
            .addHeader("Authorization", "Bearer " + getAuthToken())
            .addHeader("Content-Type", "application/json")
            .build();
        
        client.newCall(request).enqueue(callback);
    }
    
    // POST request with form data
    public void performPostWithForm(String url, Map<String, String> formData, Callback callback) {
        FormBody.Builder formBuilder = new FormBody.Builder();
        for (Map.Entry<String, String> entry : formData.entrySet()) {
            formBuilder.add(entry.getKey(), entry.getValue());
        }
        
        RequestBody formBody = formBuilder.build();
        
        Request request = new Request.Builder()
            .url(url)
            .post(formBody)
            .build();
        
        client.newCall(request).enqueue(callback);
    }
    
    // Upload file
    public void uploadFile(String url, File file, Callback callback) {
        RequestBody fileBody = RequestBody.create(file, 
            MediaType.parse("application/octet-stream"));
        
        MultipartBody requestBody = new MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", file.getName(), fileBody)
            .addFormDataPart("description", "File upload")
            .build();
        
        Request request = new Request.Builder()
            .url(url)
            .post(requestBody)
            .build();
        
        client.newCall(request).enqueue(callback);
    }
    
    private String getAuthToken() {
        // Return your auth token
        return "your_auth_token_here";
    }
}
```

### Usage Example
```java
public class ApiClient {
    private OkHttpManager httpManager;
    
    public ApiClient() {
        httpManager = new OkHttpManager();
    }
    
    public void fetchUsers() {
        httpManager.performGet("https://api.example.com/users", new Callback() {
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                // Handle network error
                Log.e("API", "Network error", e);
                
                // Switch to main thread for UI updates
                new Handler(Looper.getMainLooper()).post(() -> {
                    showError("Network error: " + e.getMessage());
                });
            }
            
            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    String responseBody = response.body().string();
                    
                    // Switch to main thread for UI updates
                    new Handler(Looper.getMainLooper()).post(() -> {
                        handleUsersResponse(responseBody);
                    });
                } else {
                    // Handle HTTP error
                    new Handler(Looper.getMainLooper()).post(() -> {
                        showError("HTTP error: " + response.code());
                    });
                }
            }
        });
    }
    
    public void createUser(User user) {
        Gson gson = new Gson();
        String json = gson.toJson(user);
        
        httpManager.performPost("https://api.example.com/users", json, new Callback() {
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    showError("Failed to create user: " + e.getMessage());
                });
            }
            
            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                new Handler(Looper.getMainLooper()).post(() -> {
                    if (response.isSuccessful()) {
                        showSuccess("User created successfully");
                    } else {
                        showError("Failed to create user: " + response.code());
                    }
                });
            }
        });
    }
}
```

## Retrofit Library

### Add Dependencies
```gradle
// app/build.gradle
dependencies {
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.11.0'
}
```

### Data Models
```java
// User model
public class User {
    @SerializedName("id")
    private int id;
    
    @SerializedName("name")
    private String name;
    
    @SerializedName("email")
    private String email;
    
    @SerializedName("phone")
    private String phone;
    
    @SerializedName("avatar_url")
    private String avatarUrl;
    
    // Constructors
    public User() {}
    
    public User(String name, String email, String phone) {
        this.name = name;
        this.email = email;
        this.phone = phone;
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
    
    public String getAvatarUrl() { return avatarUrl; }
    public void setAvatarUrl(String avatarUrl) { this.avatarUrl = avatarUrl; }
}

// API response wrapper
public class ApiResponse<T> {
    @SerializedName("success")
    private boolean success;
    
    @SerializedName("message")
    private String message;
    
    @SerializedName("data")
    private T data;
    
    @SerializedName("error")
    private String error;
    
    // Getters and setters
    public boolean isSuccess() { return success; }
    public void setSuccess(boolean success) { this.success = success; }
    
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
    
    public T getData() { return data; }
    public void setData(T data) { this.data = data; }
    
    public String getError() { return error; }
    public void setError(String error) { this.error = error; }
}

// List response
public class UserListResponse {
    @SerializedName("users")
    private List<User> users;
    
    @SerializedName("total")
    private int total;
    
    @SerializedName("page")
    private int page;
    
    @SerializedName("per_page")
    private int perPage;
    
    // Getters and setters
    public List<User> getUsers() { return users; }
    public void setUsers(List<User> users) { this.users = users; }
    
    public int getTotal() { return total; }
    public void setTotal(int total) { this.total = total; }
    
    public int getPage() { return page; }
    public void setPage(int page) { this.page = page; }
    
    public int getPerPage() { return perPage; }
    public void setPerPage(int perPage) { this.perPage = perPage; }
}
```

### API Interface
```java
public interface ApiService {
    
    // GET request
    @GET("users")
    Call<UserListResponse> getUsers();
    
    // GET with query parameters
    @GET("users")
    Call<UserListResponse> getUsers(@Query("page") int page, 
                                   @Query("per_page") int perPage);
    
    // GET with path parameter
    @GET("users/{id}")
    Call<ApiResponse<User>> getUser(@Path("id") int userId);
    
    // POST request
    @POST("users")
    Call<ApiResponse<User>> createUser(@Body User user);
    
    // PUT request
    @PUT("users/{id}")
    Call<ApiResponse<User>> updateUser(@Path("id") int userId, @Body User user);
    
    // DELETE request
    @DELETE("users/{id}")
    Call<ApiResponse<Void>> deleteUser(@Path("id") int userId);
    
    // POST with form data
    @FormUrlEncoded
    @POST("auth/login")
    Call<ApiResponse<String>> login(@Field("email") String email, 
                                   @Field("password") String password);
    
    // File upload
    @Multipart
    @POST("users/avatar")
    Call<ApiResponse<String>> uploadAvatar(@Part("user_id") RequestBody userId,
                                          @Part MultipartBody.Part file);
    
    // Download file
    @GET("files/{filename}")
    @Streaming
    Call<ResponseBody> downloadFile(@Path("filename") String filename);
    
    // Custom headers
    @GET("users")
    Call<UserListResponse> getUsersWithAuth(@Header("Authorization") String authToken);
    
    // Multiple headers
    @Headers({
        "Accept: application/json",
        "User-Agent: MyApp/1.0"
    })
    @GET("users")
    Call<UserListResponse> getUsersWithHeaders();
}
```

### Retrofit Client Setup
```java
public class RetrofitClient {
    
    private static final String BASE_URL = "https://api.example.com/";
    private static RetrofitClient instance;
    private Retrofit retrofit;
    private ApiService apiService;
    
    private RetrofitClient() {
        // Create logging interceptor
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);
        
        // Create auth interceptor
        Interceptor authInterceptor = new Interceptor() {
            @Override
            public Response intercept(Chain chain) throws IOException {
                Request originalRequest = chain.request();
                
                // Add auth header to all requests
                Request.Builder requestBuilder = originalRequest.newBuilder()
                    .addHeader("Authorization", "Bearer " + getAuthToken())
                    .addHeader("Accept", "application/json");
                
                Request newRequest = requestBuilder.build();
                return chain.proceed(newRequest);
            }
        };
        
        // Build OkHttp client
        OkHttpClient okHttpClient = new OkHttpClient.Builder()
            .addInterceptor(authInterceptor)
            .addInterceptor(logging)
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();
        
        // Build Retrofit
        retrofit = new Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build();
        
        apiService = retrofit.create(ApiService.class);
    }
    
    public static synchronized RetrofitClient getInstance() {
        if (instance == null) {
            instance = new RetrofitClient();
        }
        return instance;
    }
    
    public ApiService getApiService() {
        return apiService;
    }
    
    private String getAuthToken() {
        // Get auth token from SharedPreferences or other storage
        return "your_auth_token_here";
    }
}
```

### Repository Pattern with Retrofit
```java
public class UserRepository {
    
    private ApiService apiService;
    private MutableLiveData<List<User>> usersLiveData;
    private MutableLiveData<String> errorLiveData;
    
    public UserRepository() {
        apiService = RetrofitClient.getInstance().getApiService();
        usersLiveData = new MutableLiveData<>();
        errorLiveData = new MutableLiveData<>();
    }
    
    public LiveData<List<User>> getUsersLiveData() {
        return usersLiveData;
    }
    
    public LiveData<String> getErrorLiveData() {
        return errorLiveData;
    }
    
    public void fetchUsers() {
        Call<UserListResponse> call = apiService.getUsers();
        
        call.enqueue(new Callback<UserListResponse>() {
            @Override
            public void onResponse(Call<UserListResponse> call, Response<UserListResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    usersLiveData.setValue(response.body().getUsers());
                } else {
                    errorLiveData.setValue("Error: " + response.code());
                }
            }
            
            @Override
            public void onFailure(Call<UserListResponse> call, Throwable t) {
                errorLiveData.setValue("Network error: " + t.getMessage());
            }
        });
    }
    
    public void createUser(User user, OnResultListener<User> listener) {
        Call<ApiResponse<User>> call = apiService.createUser(user);
        
        call.enqueue(new Callback<ApiResponse<User>>() {
            @Override
            public void onResponse(Call<ApiResponse<User>> call, Response<ApiResponse<User>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    ApiResponse<User> apiResponse = response.body();
                    if (apiResponse.isSuccess()) {
                        listener.onSuccess(apiResponse.getData());
                    } else {
                        listener.onError(apiResponse.getError());
                    }
                } else {
                    listener.onError("HTTP Error: " + response.code());
                }
            }
            
            @Override
            public void onFailure(Call<ApiResponse<User>> call, Throwable t) {
                listener.onError("Network error: " + t.getMessage());
            }
        });
    }
    
    public void uploadAvatar(int userId, File imageFile, OnResultListener<String> listener) {
        // Create RequestBody for user ID
        RequestBody userIdBody = RequestBody.create(String.valueOf(userId), 
            MediaType.parse("text/plain"));
        
        // Create RequestBody for file
        RequestBody fileBody = RequestBody.create(imageFile, 
            MediaType.parse("image/*"));
        
        // Create MultipartBody.Part
        MultipartBody.Part filePart = MultipartBody.Part.createFormData(
            "avatar", imageFile.getName(), fileBody);
        
        Call<ApiResponse<String>> call = apiService.uploadAvatar(userIdBody, filePart);
        
        call.enqueue(new Callback<ApiResponse<String>>() {
            @Override
            public void onResponse(Call<ApiResponse<String>> call, Response<ApiResponse<String>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    ApiResponse<String> apiResponse = response.body();
                    if (apiResponse.isSuccess()) {
                        listener.onSuccess(apiResponse.getData());
                    } else {
                        listener.onError(apiResponse.getError());
                    }
                } else {
                    listener.onError("Upload failed: " + response.code());
                }
            }
            
            @Override
            public void onFailure(Call<ApiResponse<String>> call, Throwable t) {
                listener.onError("Upload error: " + t.getMessage());
            }
        });
    }
    
    public interface OnResultListener<T> {
        void onSuccess(T result);
        void onError(String error);
    }
}
```

### ViewModel with Repository
```java
public class UserViewModel extends ViewModel {
    
    private UserRepository repository;
    private LiveData<List<User>> users;
    private LiveData<String> error;
    
    public UserViewModel() {
        repository = new UserRepository();
        users = repository.getUsersLiveData();
        error = repository.getErrorLiveData();
    }
    
    public LiveData<List<User>> getUsers() {
        return users;
    }
    
    public LiveData<String> getError() {
        return error;
    }
    
    public void loadUsers() {
        repository.fetchUsers();
    }
    
    public void createUser(User user) {
        repository.createUser(user, new UserRepository.OnResultListener<User>() {
            @Override
            public void onSuccess(User result) {
                // Refresh users list
                repository.fetchUsers();
            }
            
            @Override
            public void onError(String error) {
                // Handle error
                Log.e("ViewModel", "Error creating user: " + error);
            }
        });
    }
}
```

## JSON Parsing

### Using Gson
```java
public class JsonParser {
    
    private Gson gson;
    
    public JsonParser() {
        gson = new GsonBuilder()
            .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
            .setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
            .create();
    }
    
    // Parse single object
    public User parseUser(String json) {
        try {
            return gson.fromJson(json, User.class);
        } catch (JsonSyntaxException e) {
            Log.e("JsonParser", "Error parsing user", e);
            return null;
        }
    }
    
    // Parse array
    public List<User> parseUserList(String json) {
        try {
            Type listType = new TypeToken<List<User>>(){}.getType();
            return gson.fromJson(json, listType);
        } catch (JsonSyntaxException e) {
            Log.e("JsonParser", "Error parsing user list", e);
            return new ArrayList<>();
        }
    }
    
    // Convert object to JSON
    public String userToJson(User user) {
        return gson.toJson(user);
    }
    
    // Parse nested JSON
    public class ApiResponse<T> {
        private boolean success;
        private String message;
        private T data;
        
        // Getters and setters...
    }
    
    public ApiResponse<User> parseApiResponse(String json) {
        try {
            Type responseType = new TypeToken<ApiResponse<User>>(){}.getType();
            return gson.fromJson(json, responseType);
        } catch (JsonSyntaxException e) {
            Log.e("JsonParser", "Error parsing API response", e);
            return null;
        }
    }
}
```

### Manual JSON Parsing
```java
public class ManualJsonParser {
    
    public static User parseUser(String json) {
        try {
            JSONObject jsonObject = new JSONObject(json);
            
            User user = new User();
            user.setId(jsonObject.getInt("id"));
            user.setName(jsonObject.getString("name"));
            user.setEmail(jsonObject.getString("email"));
            
            // Handle optional fields
            if (jsonObject.has("phone")) {
                user.setPhone(jsonObject.getString("phone"));
            }
            
            return user;
        } catch (JSONException e) {
            Log.e("JsonParser", "Error parsing user", e);
            return null;
        }
    }
    
    public static List<User> parseUserList(String json) {
        List<User> users = new ArrayList<>();
        
        try {
            JSONArray jsonArray = new JSONArray(json);
            
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject userJson = jsonArray.getJSONObject(i);
                User user = parseUser(userJson.toString());
                if (user != null) {
                    users.add(user);
                }
            }
        } catch (JSONException e) {
            Log.e("JsonParser", "Error parsing user list", e);
        }
        
        return users;
    }
    
    public static String userToJson(User user) {
        try {
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("name", user.getName());
            jsonObject.put("email", user.getEmail());
            jsonObject.put("phone", user.getPhone());
            
            return jsonObject.toString();
        } catch (JSONException e) {
            Log.e("JsonParser", "Error creating JSON", e);
            return null;
        }
    }
}
```

## Image Loading

### Using Glide
```gradle
// app/build.gradle
dependencies {
    implementation 'com.github.bumptech.glide:glide:4.15.1'
    annotationProcessor 'com.github.bumptech.glide:compiler:4.15.1'
}
```

```java
public class ImageLoader {
    
    // Basic image loading
    public static void loadImage(Context context, String url, ImageView imageView) {
        Glide.with(context)
            .load(url)
            .into(imageView);
    }
    
    // Image loading with placeholder and error handling
    public static void loadImageWithPlaceholder(Context context, String url, ImageView imageView) {
        Glide.with(context)
            .load(url)
            .placeholder(R.drawable.placeholder)
            .error(R.drawable.error_image)
            .into(imageView);
    }
    
    // Circular image
    public static void loadCircularImage(Context context, String url, ImageView imageView) {
        Glide.with(context)
            .load(url)
            .circleCrop()
            .placeholder(R.drawable.placeholder_circle)
            .into(imageView);
    }
    
    // Rounded corners
    public static void loadRoundedImage(Context context, String url, ImageView imageView, int radius) {
        Glide.with(context)
            .load(url)
            .transform(new RoundedCorners(radius))
            .into(imageView);
    }
    
    // Custom size
    public static void loadImageWithSize(Context context, String url, ImageView imageView, 
                                       int width, int height) {
        Glide.with(context)
            .load(url)
            .override(width, height)
            .into(imageView);
    }
    
    // Cache control
    public static void loadImageWithCache(Context context, String url, ImageView imageView) {
        Glide.with(context)
            .load(url)
            .diskCacheStrategy(DiskCacheStrategy.ALL)
            .skipMemoryCache(false)
            .into(imageView);
    }
    
    // Load from file
    public static void loadImageFromFile(Context context, File file, ImageView imageView) {
        Glide.with(context)
            .load(file)
            .into(imageView);
    }
    
    // Load from URI
    public static void loadImageFromUri(Context context, Uri uri, ImageView imageView) {
        Glide.with(context)
            .load(uri)
            .into(imageView);
    }
}
```

### Using Picasso
```gradle
// app/build.gradle
dependencies {
    implementation 'com.squareup.picasso:picasso:2.8'
}
```

```java
public class PicassoImageLoader {
    
    public static void loadImage(String url, ImageView imageView) {
        Picasso.get()
            .load(url)
            .placeholder(R.drawable.placeholder)
            .error(R.drawable.error_image)
            .into(imageView);
    }
    
    public static void loadCircularImage(String url, ImageView imageView) {
        Picasso.get()
            .load(url)
            .transform(new CircleTransform())
            .into(imageView);
    }
    
    // Custom transformation for circle
    public static class CircleTransform implements Transformation {
        @Override
        public Bitmap transform(Bitmap source) {
            int size = Math.min(source.getWidth(), source.getHeight());
            
            int x = (source.getWidth() - size) / 2;
            int y = (source.getHeight() - size) / 2;
            
            Bitmap squaredBitmap = Bitmap.createBitmap(source, x, y, size, size);
            if (squaredBitmap != source) {
                source.recycle();
            }
            
            Bitmap bitmap = Bitmap.createBitmap(size, size, source.getConfig());
            
            Canvas canvas = new Canvas(bitmap);
            Paint paint = new Paint();
            BitmapShader shader = new BitmapShader(squaredBitmap, 
                BitmapShader.TileMode.CLAMP, BitmapShader.TileMode.CLAMP);
            paint.setShader(shader);
            paint.setAntiAlias(true);
            
            float radius = size / 2f;
            canvas.drawCircle(radius, radius, radius, paint);
            
            squaredBitmap.recycle();
            return bitmap;
        }
        
        @Override
        public String key() {
            return "circle";
        }
    }
}
```

## Network Security

### Certificate Pinning
```java
public class SecureHttpClient {
    
    public static OkHttpClient createSecureClient() {
        // Certificate pinning
        CertificatePinner certificatePinner = new CertificatePinner.Builder()
            .add("api.example.com", "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
            .add("api.example.com", "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=")
            .build();
        
        return new OkHttpClient.Builder()
            .certificatePinner(certificatePinner)
            .build();
    }
    
    // Custom trust manager (use with caution)
    public static OkHttpClient createTrustAllClient() {
        try {
            final TrustManager[] trustAllCerts = new TrustManager[] {
                new X509TrustManager() {
                    @Override
                    public void checkClientTrusted(X509Certificate[] chain, String authType) {}
                    
                    @Override
                    public void checkServerTrusted(X509Certificate[] chain, String authType) {}
                    
                    @Override
                    public X509Certificate[] getAcceptedIssuers() {
                        return new X509Certificate[]{};
                    }
                }
            };
            
            final SSLContext sslContext = SSLContext.getInstance("SSL");
            sslContext.init(null, trustAllCerts, new java.security.SecureRandom());
            
            final SSLSocketFactory sslSocketFactory = sslContext.getSocketFactory();
            
            return new OkHttpClient.Builder()
                .sslSocketFactory(sslSocketFactory, (X509TrustManager)trustAllCerts[0])
                .hostnameVerifier((hostname, session) -> true)
                .build();
                
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
```

## Error Handling

### Comprehensive Error Handling
```java
public class ApiErrorHandler {
    
    public static class ApiError {
        private int code;
        private String message;
        private String details;
        
        public ApiError(int code, String message, String details) {
            this.code = code;
            this.message = message;
            this.details = details;
        }
        
        // Getters
        public int getCode() { return code; }
        public String getMessage() { return message; }
        public String getDetails() { return details; }
    }
    
    public static ApiError handleError(Response<?> response) {
        int code = response.code();
        String message;
        String details = "";
        
        switch (code) {
            case 400:
                message = "Bad Request";
                break;
            case 401:
                message = "Unauthorized";
                break;
            case 403:
                message = "Forbidden";
                break;
            case 404:
                message = "Not Found";
                break;
            case 500:
                message = "Internal Server Error";
                break;
            case 502:
                message = "Bad Gateway";
                break;
            case 503:
                message = "Service Unavailable";
                break;
            default:
                message = "Unknown Error";
        }
        
        // Try to get error details from response body
        try {
            if (response.errorBody() != null) {
                String errorBody = response.errorBody().string();
                JSONObject errorJson = new JSONObject(errorBody);
                if (errorJson.has("message")) {
                    details = errorJson.getString("message");
                }
            }
        } catch (Exception e) {
            Log.e("ApiErrorHandler", "Error parsing error response", e);
        }
        
        return new ApiError(code, message, details);
    }
    
    public static ApiError handleException(Throwable throwable) {
        if (throwable instanceof UnknownHostException) {
            return new ApiError(-1, "No Internet Connection", 
                "Please check your internet connection and try again.");
        } else if (throwable instanceof SocketTimeoutException) {
            return new ApiError(-2, "Request Timeout", 
                "The request took too long to complete.");
        } else if (throwable instanceof ConnectException) {
            return new ApiError(-3, "Connection Failed", 
                "Unable to connect to the server.");
        } else {
            return new ApiError(-4, "Network Error", 
                throwable.getMessage());
        }
    }
}
```

## Best Practices

### 1. Use Repository Pattern
```java
public abstract class BaseRepository {
    
    protected <T> void handleResponse(Response<T> response, 
                                    OnResultListener<T> listener) {
        if (response.isSuccessful() && response.body() != null) {
            listener.onSuccess(response.body());
        } else {
            ApiErrorHandler.ApiError error = ApiErrorHandler.handleError(response);
            listener.onError(error.getMessage());
        }
    }
    
    protected <T> void handleFailure(Throwable throwable, 
                                   OnResultListener<T> listener) {
        ApiErrorHandler.ApiError error = ApiErrorHandler.handleException(throwable);
        listener.onError(error.getMessage());
    }
    
    public interface OnResultListener<T> {
        void onSuccess(T result);
        void onError(String error);
    }
}
```

### 2. Implement Retry Logic
```java
public class RetryInterceptor implements Interceptor {
    
    private int maxRetry = 3;
    private int retryInterval = 1000; // 1 second
    
    @Override
    public Response intercept(Chain chain) throws IOException {
        Request request = chain.request();
        Response response = null;
        IOException exception = null;
        
        for (int attempt = 0; attempt < maxRetry; attempt++) {
            try {
                response = chain.proceed(request);
                
                if (response.isSuccessful()) {
                    return response;
                }
                
                // Retry on 5xx errors
                if (response.code() >= 500) {
                    if (attempt < maxRetry - 1) {
                        response.close();
                        Thread.sleep(retryInterval);
                        continue;
                    }
                }
                
                return response;
                
            } catch (IOException e) {
                exception = e;
                if (attempt < maxRetry - 1) {
                    try {
                        Thread.sleep(retryInterval);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        
        if (response != null) {
            return response;
        } else {
            throw exception;
        }
    }
}
```

### 3. Cache Management
```java
public class CacheInterceptor implements Interceptor {
    
    @Override
    public Response intercept(Chain chain) throws IOException {
        Request request = chain.request();
        
        // Check if network is available
        if (isNetworkAvailable()) {
            // If network is available, get fresh data
            request = request.newBuilder()
                .cacheControl(new CacheControl.Builder()
                    .maxAge(60, TimeUnit.SECONDS)
                    .build())
                .build();
        } else {
            // If no network, use cached data
            request = request.newBuilder()
                .cacheControl(new CacheControl.Builder()
                    .maxStale(7, TimeUnit.DAYS)
                    .onlyIfCached()
                    .build())
                .build();
        }
        
        return chain.proceed(request);
    }
    
    private boolean isNetworkAvailable() {
        // Implement network check
        return true;
    }
}
```

### 4. Request/Response Logging
```java
public class CustomLoggingInterceptor implements Interceptor {
    
    @Override
    public Response intercept(Chain chain) throws IOException {
        Request request = chain.request();
        
        // Log request
        Log.d("HTTP", "Request: " + request.method() + " " + request.url());
        Log.d("HTTP", "Headers: " + request.headers());
        
        long startTime = System.nanoTime();
        Response response = chain.proceed(request);
        long endTime = System.nanoTime();
        
        // Log response
        Log.d("HTTP", "Response: " + response.code() + " in " + 
            (endTime - startTime) / 1e6d + "ms");
        
        return response;
    }
}
```

### 5. Thread Management
```java
public class ApiManager {
    
    private ExecutorService executorService;
    private Handler mainHandler;
    
    public ApiManager() {
        executorService = Executors.newFixedThreadPool(4);
        mainHandler = new Handler(Looper.getMainLooper());
    }
    
    public <T> void executeAsync(Callable<T> task, OnResultListener<T> listener) {
        executorService.execute(() -> {
            try {
                T result = task.call();
                mainHandler.post(() -> listener.onSuccess(result));
            } catch (Exception e) {
                mainHandler.post(() -> listener.onError(e.getMessage()));
            }
        });
    }
    
    public void shutdown() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
    }
}
```

Understanding networking and API integration is crucial for modern Android applications. Use established libraries like Retrofit and OkHttp for robust, maintainable networking code, and always implement proper error handling and security measures.
