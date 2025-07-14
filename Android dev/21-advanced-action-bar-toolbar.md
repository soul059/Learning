# Advanced Action Bar and Toolbar

## Table of Contents
- [Overview](#overview)
- [Setting Up Toolbar](#setting-up-toolbar)
- [Customizing Action Bar](#customizing-action-bar)
- [Action Bar Menus](#action-bar-menus)
- [SearchView Integration](#searchview-integration)
- [Navigation Drawer with Action Bar](#navigation-drawer-with-action-bar)
- [Up Navigation](#up-navigation)
- [Action Bar Themes and Styling](#action-bar-themes-and-styling)
- [Multiple Action Bars](#multiple-action-bars)
- [Best Practices](#best-practices)

## Overview

The Action Bar (now typically implemented as Toolbar) is a key UI component that provides navigation, branding, and actions for your app.

### Key Components
- **App icon/logo**: Branding and navigation
- **Title**: Current screen or app name
- **Action items**: Primary actions for current screen
- **Overflow menu**: Secondary actions
- **Navigation button**: Up/back navigation

### Benefits of Toolbar vs ActionBar
- **Flexibility**: Can be placed anywhere in layout
- **Customization**: Full control over appearance
- **Material Design**: Better support for Material Design
- **Animations**: Easier to animate and transform

## Setting Up Toolbar

### Basic Toolbar Setup
```xml
<!-- activity_main.xml -->
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical">

    <androidx.appcompat.widget.Toolbar
        android:id="@+id/toolbar"
        android:layout_width="match_parent"
        android:layout_height="?attr/actionBarSize"
        android:background="?attr/colorPrimary"
        android:elevation="4dp"
        android:theme="@style/ThemeOverlay.AppCompat.ActionBar"
        app:popupTheme="@style/ThemeOverlay.AppCompat.Light" />

    <FrameLayout
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1">
        
        <!-- Your content here -->
        
    </FrameLayout>

</LinearLayout>
```

### Activity Implementation
```java
public class MainActivity extends AppCompatActivity {

    private Toolbar toolbar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setupToolbar();
    }

    private void setupToolbar() {
        toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        // Enable up button
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
            actionBar.setDisplayShowHomeEnabled(true);
            actionBar.setTitle("My App");
            actionBar.setSubtitle("Home");
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                onBackPressed();
                return true;
            case R.id.action_settings:
                // Handle settings action
                return true;
            case R.id.action_search:
                // Handle search action
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }
}
```

### Theme Configuration
```xml
<!-- styles.xml -->
<resources>
    <!-- Base application theme -->
    <style name="AppTheme" parent="Theme.AppCompat.Light.NoActionBar">
        <item name="colorPrimary">@color/colorPrimary</item>
        <item name="colorPrimaryDark">@color/colorPrimaryDark</item>
        <item name="colorAccent">@color/colorAccent</item>
    </style>

    <!-- Toolbar theme -->
    <style name="ToolbarTheme" parent="ThemeOverlay.AppCompat.ActionBar">
        <item name="android:textColorPrimary">@android:color/white</item>
        <item name="android:textColorSecondary">@android:color/white</item>
        <item name="actionMenuTextColor">@android:color/white</item>
    </style>

    <!-- Popup theme for overflow menu -->
    <style name="ToolbarPopupTheme" parent="ThemeOverlay.AppCompat.Light">
        <item name="android:textColor">@android:color/black</item>
    </style>
</resources>
```

## Customizing Action Bar

### Custom Toolbar Layout
```xml
<!-- custom_toolbar.xml -->
<androidx.appcompat.widget.Toolbar xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:id="@+id/toolbar"
    android:layout_width="match_parent"
    android:layout_height="?attr/actionBarSize"
    android:background="?attr/colorPrimary"
    android:elevation="4dp"
    android:theme="@style/ToolbarTheme"
    app:popupTheme="@style/ToolbarPopupTheme">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:gravity="center_vertical"
        android:orientation="horizontal">

        <ImageView
            android:id="@+id/toolbar_logo"
            android:layout_width="40dp"
            android:layout_height="40dp"
            android:layout_marginEnd="16dp"
            android:src="@drawable/ic_app_logo"
            android:contentDescription="App Logo" />

        <LinearLayout
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:orientation="vertical">

            <TextView
                android:id="@+id/toolbar_title"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Custom Title"
                android:textColor="@android:color/white"
                android:textSize="18sp"
                android:textStyle="bold" />

            <TextView
                android:id="@+id/toolbar_subtitle"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Subtitle"
                android:textColor="@android:color/white"
                android:textSize="14sp"
                android:alpha="0.8" />

        </LinearLayout>

        <ImageButton
            android:id="@+id/toolbar_notification"
            android:layout_width="48dp"
            android:layout_height="48dp"
            android:background="?attr/selectableItemBackgroundBorderless"
            android:src="@drawable/ic_notifications"
            android:contentDescription="Notifications" />

    </LinearLayout>

</androidx.appcompat.widget.Toolbar>
```

### Advanced Toolbar Customization
```java
public class CustomToolbarActivity extends AppCompatActivity {

    private Toolbar toolbar;
    private TextView toolbarTitle, toolbarSubtitle;
    private ImageView toolbarLogo;
    private ImageButton notificationButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_custom_toolbar);

        setupCustomToolbar();
    }

    private void setupCustomToolbar() {
        toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        // Disable default title
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayShowTitleEnabled(false);
        }

        // Initialize custom views
        toolbarTitle = findViewById(R.id.toolbar_title);
        toolbarSubtitle = findViewById(R.id.toolbar_subtitle);
        toolbarLogo = findViewById(R.id.toolbar_logo);
        notificationButton = findViewById(R.id.toolbar_notification);

        // Setup click listeners
        toolbarLogo.setOnClickListener(v -> {
            // Handle logo click
            Toast.makeText(this, "Logo clicked", Toast.LENGTH_SHORT).show();
        });

        notificationButton.setOnClickListener(v -> {
            // Handle notification click
            showNotifications();
        });

        // Setup toolbar navigation
        toolbar.setNavigationIcon(R.drawable.ic_arrow_back);
        toolbar.setNavigationOnClickListener(v -> onBackPressed());
    }

    public void updateToolbarTitle(String title, String subtitle) {
        toolbarTitle.setText(title);
        toolbarSubtitle.setText(subtitle);
    }

    public void setNotificationCount(int count) {
        if (count > 0) {
            // Show badge on notification icon
            showNotificationBadge(count);
        } else {
            hideNotificationBadge();
        }
    }

    private void showNotificationBadge(int count) {
        // Implementation for notification badge
        // You can use a library like BadgeDrawable or create custom badge
    }

    private void hideNotificationBadge() {
        // Hide notification badge
    }

    private void showNotifications() {
        // Show notifications screen or popup
        Intent intent = new Intent(this, NotificationsActivity.class);
        startActivity(intent);
    }
}
```

## Action Bar Menus

### Menu Resource Definition
```xml
<!-- menu/main_menu.xml -->
<menu xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto">

    <item
        android:id="@+id/action_search"
        android:icon="@drawable/ic_search"
        android:title="Search"
        app:showAsAction="ifRoom" />

    <item
        android:id="@+id/action_favorite"
        android:icon="@drawable/ic_favorite"
        android:title="Favorite"
        app:showAsAction="ifRoom" />

    <item
        android:id="@+id/action_share"
        android:icon="@drawable/ic_share"
        android:title="Share"
        app:showAsAction="ifRoom" />

    <item
        android:id="@+id/action_settings"
        android:icon="@drawable/ic_settings"
        android:title="Settings"
        app:showAsAction="never" />

    <item
        android:id="@+id/action_about"
        android:title="About"
        app:showAsAction="never" />

    <item
        android:id="@+id/action_help"
        android:title="Help"
        app:showAsAction="never" />

</menu>
```

### Dynamic Menu Management
```java
public class MenuActivity extends AppCompatActivity {

    private Menu optionsMenu;
    private boolean isFavorite = false;

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        this.optionsMenu = menu;
        
        updateMenuState();
        return true;
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        // Called every time menu is shown
        updateMenuState();
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_search:
                handleSearchAction();
                return true;
                
            case R.id.action_favorite:
                toggleFavorite();
                return true;
                
            case R.id.action_share:
                handleShareAction();
                return true;
                
            case R.id.action_settings:
                openSettings();
                return true;
                
            case R.id.action_about:
                showAbout();
                return true;
                
            case R.id.action_help:
                showHelp();
                return true;
                
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    private void updateMenuState() {
        if (optionsMenu != null) {
            MenuItem favoriteItem = optionsMenu.findItem(R.id.action_favorite);
            if (favoriteItem != null) {
                favoriteItem.setIcon(isFavorite ? 
                    R.drawable.ic_favorite_filled : R.drawable.ic_favorite_border);
                favoriteItem.setTitle(isFavorite ? "Remove Favorite" : "Add Favorite");
            }
            
            // Show/hide menu items based on conditions
            MenuItem shareItem = optionsMenu.findItem(R.id.action_share);
            if (shareItem != null) {
                shareItem.setVisible(hasContentToShare());
            }
        }
    }

    private void toggleFavorite() {
        isFavorite = !isFavorite;
        updateMenuState();
        
        String message = isFavorite ? "Added to favorites" : "Removed from favorites";
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

    private void handleSearchAction() {
        // Implement search functionality
        Intent searchIntent = new Intent(this, SearchActivity.class);
        startActivity(searchIntent);
    }

    private void handleShareAction() {
        Intent shareIntent = new Intent(Intent.ACTION_SEND);
        shareIntent.setType("text/plain");
        shareIntent.putExtra(Intent.EXTRA_TEXT, "Check out this amazing app!");
        shareIntent.putExtra(Intent.EXTRA_SUBJECT, "App Recommendation");
        startActivity(Intent.createChooser(shareIntent, "Share via"));
    }

    private void openSettings() {
        Intent settingsIntent = new Intent(this, SettingsActivity.class);
        startActivity(settingsIntent);
    }

    private void showAbout() {
        // Show about dialog or activity
        AboutDialogFragment.newInstance().show(getSupportFragmentManager(), "about");
    }

    private void showHelp() {
        // Show help screen
        Intent helpIntent = new Intent(this, HelpActivity.class);
        startActivity(helpIntent);
    }

    private boolean hasContentToShare() {
        // Logic to determine if there's content to share
        return true;
    }
}
```

## SearchView Integration

### SearchView in Action Bar
```xml
<!-- menu/search_menu.xml -->
<menu xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto">

    <item
        android:id="@+id/action_search"
        android:icon="@drawable/ic_search"
        android:title="Search"
        app:showAsAction="ifRoom|collapseActionView"
        app:actionViewClass="androidx.appcompat.widget.SearchView" />

</menu>
```

### SearchView Implementation
```java
public class SearchActivity extends AppCompatActivity {

    private SearchView searchView;
    private RecyclerView recyclerView;
    private SearchAdapter adapter;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_search);
        
        setupToolbar();
        setupRecyclerView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.search_menu, menu);
        
        MenuItem searchItem = menu.findItem(R.id.action_search);
        searchView = (SearchView) searchItem.getActionView();
        
        setupSearchView(searchView, searchItem);
        
        return true;
    }

    private void setupSearchView(SearchView searchView, MenuItem searchItem) {
        searchView.setQueryHint("Search items...");
        searchView.setMaxWidth(Integer.MAX_VALUE);
        
        // Handle search query text changes
        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
                performSearch(query);
                searchView.clearFocus();
                return true;
            }

            @Override
            public boolean onQueryTextChange(String newText) {
                // Filter results as user types
                filterResults(newText);
                return true;
            }
        });

        // Handle search view expand/collapse
        searchItem.setOnActionExpandListener(new MenuItem.OnActionExpandListener() {
            @Override
            public boolean onMenuItemActionExpand(MenuItem item) {
                // Search expanded
                return true;
            }

            @Override
            public boolean onMenuItemActionCollapse(MenuItem item) {
                // Search collapsed - restore original list
                adapter.resetFilter();
                return true;
            }
        });

        // Auto-expand search if this is a search-focused activity
        searchItem.expandActionView();
    }

    private void performSearch(String query) {
        // Perform search operation
        List<SearchResult> results = searchInDatabase(query);
        adapter.updateResults(results);
        
        // Save search query to recent searches
        saveRecentSearch(query);
    }

    private void filterResults(String query) {
        adapter.filter(query);
    }

    private List<SearchResult> searchInDatabase(String query) {
        // Implement your search logic here
        return new ArrayList<>();
    }

    private void saveRecentSearch(String query) {
        SharedPreferences prefs = getSharedPreferences("search_history", MODE_PRIVATE);
        Set<String> recentSearches = prefs.getStringSet("recent", new HashSet<>());
        
        recentSearches.add(query);
        
        // Keep only last 10 searches
        if (recentSearches.size() > 10) {
            List<String> searchList = new ArrayList<>(recentSearches);
            recentSearches.clear();
            recentSearches.addAll(searchList.subList(searchList.size() - 10, searchList.size()));
        }
        
        prefs.edit().putStringSet("recent", recentSearches).apply();
    }

    private void setupRecyclerView() {
        recyclerView = findViewById(R.id.recyclerView);
        adapter = new SearchAdapter();
        recyclerView.setAdapter(adapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
    }
}
```

## Navigation Drawer with Action Bar

### Navigation Drawer Setup
```java
public class DrawerActivity extends AppCompatActivity 
        implements NavigationView.OnNavigationItemSelectedListener {

    private DrawerLayout drawerLayout;
    private ActionBarDrawerToggle toggle;
    private NavigationView navigationView;
    private Toolbar toolbar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_drawer);

        setupToolbar();
        setupNavigationDrawer();
    }

    private void setupToolbar() {
        toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
    }

    private void setupNavigationDrawer() {
        drawerLayout = findViewById(R.id.drawer_layout);
        navigationView = findViewById(R.id.nav_view);

        // Setup drawer toggle
        toggle = new ActionBarDrawerToggle(
                this, drawerLayout, toolbar,
                R.string.navigation_drawer_open,
                R.string.navigation_drawer_close);

        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();

        // Set navigation item selected listener
        navigationView.setNavigationItemSelectedListener(this);

        // Setup header
        setupNavigationHeader();
    }

    private void setupNavigationHeader() {
        View headerView = navigationView.getHeaderView(0);
        
        TextView userName = headerView.findViewById(R.id.user_name);
        TextView userEmail = headerView.findViewById(R.id.user_email);
        ImageView userAvatar = headerView.findViewById(R.id.user_avatar);

        // Set user information
        userName.setText("John Doe");
        userEmail.setText("john.doe@example.com");
        
        // Load user avatar (using Glide or similar)
        // Glide.with(this).load(userAvatarUrl).into(userAvatar);
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem item) {
        int id = item.getItemId();

        switch (id) {
            case R.id.nav_home:
                loadFragment(new HomeFragment());
                break;
            case R.id.nav_profile:
                loadFragment(new ProfileFragment());
                break;
            case R.id.nav_settings:
                startActivity(new Intent(this, SettingsActivity.class));
                break;
            case R.id.nav_logout:
                handleLogout();
                break;
        }

        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }

    private void loadFragment(Fragment fragment) {
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.fragment_container, fragment)
                .commit();
    }

    private void handleLogout() {
        new AlertDialog.Builder(this)
                .setTitle("Logout")
                .setMessage("Are you sure you want to logout?")
                .setPositiveButton("Logout", (dialog, which) -> {
                    // Perform logout
                    finish();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    @Override
    public void onBackPressed() {
        if (drawerLayout.isDrawerOpen(GravityCompat.START)) {
            drawerLayout.closeDrawer(GravityCompat.START);
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        toggle.syncState();
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        toggle.onConfigurationChanged(newConfig);
    }
}
```

## Up Navigation

### Hierarchical Navigation Setup
```xml
<!-- AndroidManifest.xml -->
<activity
    android:name=".DetailActivity"
    android:parentActivityName=".MainActivity">
    <meta-data
        android:name="android.support.PARENT_ACTIVITY"
        android:value=".MainActivity" />
</activity>
```

### Up Navigation Implementation
```java
public class DetailActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detail);

        setupToolbar();
    }

    private void setupToolbar() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
            actionBar.setDisplayShowHomeEnabled(true);
            actionBar.setTitle("Details");
        }
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                // Handle up navigation
                onBackPressed(); // For temporal navigation
                // OR
                // NavUtils.navigateUpFromSameTask(this); // For hierarchical navigation
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    @Override
    public boolean onSupportNavigateUp() {
        onBackPressed();
        return true;
    }
}
```

## Action Bar Themes and Styling

### Custom Action Bar Styles
```xml
<!-- styles.xml -->
<resources>
    <!-- Custom ActionBar theme -->
    <style name="CustomActionBarTheme" parent="Theme.AppCompat.Light">
        <item name="colorPrimary">@color/custom_primary</item>
        <item name="colorPrimaryDark">@color/custom_primary_dark</item>
        <item name="colorAccent">@color/custom_accent</item>
        <item name="actionBarStyle">@style/CustomActionBarStyle</item>
        <item name="actionBarTheme">@style/CustomActionBarThemeOverlay</item>
    </style>

    <!-- Custom ActionBar style -->
    <style name="CustomActionBarStyle" parent="Widget.AppCompat.ActionBar">
        <item name="background">@drawable/actionbar_background</item>
        <item name="titleTextStyle">@style/CustomActionBarTitleText</item>
        <item name="subtitleTextStyle">@style/CustomActionBarSubtitleText</item>
        <item name="elevation">8dp</item>
    </style>

    <!-- ActionBar theme overlay -->
    <style name="CustomActionBarThemeOverlay" parent="ThemeOverlay.AppCompat.ActionBar">
        <item name="colorControlNormal">@android:color/white</item>
        <item name="actionMenuTextColor">@android:color/white</item>
    </style>

    <!-- Title text style -->
    <style name="CustomActionBarTitleText" parent="TextAppearance.AppCompat.Widget.ActionBar.Title">
        <item name="android:textColor">@android:color/white</item>
        <item name="android:textSize">20sp</item>
        <item name="android:textStyle">bold</item>
        <item name="android:fontFamily">@font/custom_font</item>
    </style>

    <!-- Subtitle text style -->
    <style name="CustomActionBarSubtitleText" parent="TextAppearance.AppCompat.Widget.ActionBar.Subtitle">
        <item name="android:textColor">@android:color/white</item>
        <item name="android:textSize">14sp</item>
        <item name="android:alpha">0.8</item>
    </style>
</resources>
```

### Gradient Action Bar Background
```xml
<!-- drawable/actionbar_background.xml -->
<shape xmlns:android="http://schemas.android.com/apk/res/android">
    <gradient
        android:startColor="#FF6B35"
        android:endColor="#F7931E"
        android:angle="90" />
</shape>
```

## Best Practices

### Action Bar Design Guidelines
- **Consistent branding**: Use consistent colors and fonts
- **Clear hierarchy**: Primary actions visible, secondary in overflow
- **Appropriate icons**: Use standard Android icons when possible
- **Responsive design**: Adapt to different screen sizes
- **Accessibility**: Provide content descriptions for icons

### Performance Considerations
- **Menu inflation**: Cache menu inflation when possible
- **Icon optimization**: Use vector drawables for scalability
- **Search optimization**: Implement efficient search algorithms
- **Memory management**: Release resources properly

### User Experience
- **Predictable navigation**: Follow Android navigation patterns
- **Quick access**: Place frequently used actions prominently
- **Visual feedback**: Show selected states and loading indicators
- **Contextual actions**: Show relevant actions based on current content

Understanding advanced Action Bar and Toolbar implementation enables creating professional, navigation-rich Android applications with consistent and intuitive user interfaces following Material Design principles.
