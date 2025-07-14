# Tab Navigation and ViewPager

## Table of Contents
- [Overview](#overview)
- [TabLayout Basics](#tablayout-basics)
- [ViewPager2 with Fragments](#viewpager2-with-fragments)
- [ViewPager2 with TabLayout](#viewpager2-with-tablayout)
- [Custom Tab Layouts](#custom-tab-layouts)
- [Dynamic Tab Management](#dynamic-tab-management)
- [Tab Badges and Indicators](#tab-badges-and-indicators)
- [Scrollable Tabs](#scrollable-tabs)
- [Bottom Navigation with Tabs](#bottom-navigation-with-tabs)
- [Advanced Tab Features](#advanced-tab-features)
- [Best Practices](#best-practices)

## Overview

Tab navigation provides a way to organize content into categories and allow users to quickly switch between different sections of your app.

### Key Components
- **TabLayout**: Container for tabs
- **ViewPager2**: Swipeable page container
- **FragmentStateAdapter**: Adapter for fragment-based pages
- **TabLayoutMediator**: Connects TabLayout with ViewPager2

### Tab Types
- **Fixed tabs**: All tabs visible, equal width
- **Scrollable tabs**: Tabs can scroll horizontally
- **Icon tabs**: Tabs with icons only
- **Text tabs**: Tabs with text only
- **Combined tabs**: Tabs with both text and icons

## TabLayout Basics

### Basic TabLayout Setup
```xml
<!-- activity_tabs.xml -->
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
        android:theme="@style/ThemeOverlay.AppCompat.ActionBar"
        app:popupTheme="@style/ThemeOverlay.AppCompat.Light" />

    <com.google.android.material.tabs.TabLayout
        android:id="@+id/tabLayout"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:background="?attr/colorPrimary"
        app:tabTextColor="@android:color/white"
        app:tabSelectedTextColor="@android:color/white"
        app:tabIndicatorColor="@android:color/white"
        app:tabMode="fixed"
        app:tabGravity="fill" />

    <androidx.viewpager2.widget.ViewPager2
        android:id="@+id/viewPager"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1" />

</LinearLayout>
```

### Simple Tab Activity
```java
public class TabsActivity extends AppCompatActivity {

    private TabLayout tabLayout;
    private ViewPager2 viewPager;
    private TabsPagerAdapter pagerAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tabs);

        setupToolbar();
        setupTabs();
    }

    private void setupToolbar() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setTitle("Tabs Example");
        }
    }

    private void setupTabs() {
        tabLayout = findViewById(R.id.tabLayout);
        viewPager = findViewById(R.id.viewPager);

        // Create adapter
        pagerAdapter = new TabsPagerAdapter(this);
        viewPager.setAdapter(pagerAdapter);

        // Connect TabLayout with ViewPager2
        new TabLayoutMediator(tabLayout, viewPager,
                (tab, position) -> {
                    switch (position) {
                        case 0:
                            tab.setText("Home");
                            tab.setIcon(R.drawable.ic_home);
                            break;
                        case 1:
                            tab.setText("Favorites");
                            tab.setIcon(R.drawable.ic_favorite);
                            break;
                        case 2:
                            tab.setText("Profile");
                            tab.setIcon(R.drawable.ic_person);
                            break;
                        case 3:
                            tab.setText("Settings");
                            tab.setIcon(R.drawable.ic_settings);
                            break;
                    }
                }).attach();
    }

    private static class TabsPagerAdapter extends FragmentStateAdapter {

        public TabsPagerAdapter(@NonNull FragmentActivity fragmentActivity) {
            super(fragmentActivity);
        }

        @NonNull
        @Override
        public Fragment createFragment(int position) {
            switch (position) {
                case 0:
                    return new HomeFragment();
                case 1:
                    return new FavoritesFragment();
                case 2:
                    return new ProfileFragment();
                case 3:
                    return new SettingsFragment();
                default:
                    return new HomeFragment();
            }
        }

        @Override
        public int getItemCount() {
            return 4;
        }
    }
}
```

## ViewPager2 with Fragments

### Fragment Implementation
```java
public class HomeFragment extends Fragment {

    private RecyclerView recyclerView;
    private HomeAdapter adapter;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, 
                           @Nullable ViewGroup container, 
                           @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_home, container, false);
        
        setupRecyclerView(view);
        loadData();
        
        return view;
    }

    private void setupRecyclerView(View view) {
        recyclerView = view.findViewById(R.id.recyclerView);
        adapter = new HomeAdapter();
        recyclerView.setAdapter(adapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
    }

    private void loadData() {
        // Load data for this tab
        List<HomeItem> items = getHomeItems();
        adapter.updateItems(items);
    }

    private List<HomeItem> getHomeItems() {
        // Return mock data or load from database/network
        return new ArrayList<>();
    }

    // Refresh data when tab becomes visible
    @Override
    public void setUserVisibleHint(boolean isVisibleToUser) {
        super.setUserVisibleHint(isVisibleToUser);
        
        if (isVisibleToUser && isResumed()) {
            refreshData();
        }
    }

    private void refreshData() {
        // Refresh data when tab becomes active
        loadData();
    }
}
```

### Advanced ViewPager2 Adapter
```java
public class AdvancedTabsPagerAdapter extends FragmentStateAdapter {

    private final List<TabInfo> tabInfoList;

    public static class TabInfo {
        public String title;
        public int iconResId;
        public Fragment fragment;
        public int badgeCount = 0;

        public TabInfo(String title, int iconResId, Fragment fragment) {
            this.title = title;
            this.iconResId = iconResId;
            this.fragment = fragment;
        }
    }

    public AdvancedTabsPagerAdapter(@NonNull FragmentActivity fragmentActivity) {
        super(fragmentActivity);
        this.tabInfoList = new ArrayList<>();
        
        initializeTabs();
    }

    private void initializeTabs() {
        addTab("Home", R.drawable.ic_home, new HomeFragment());
        addTab("Search", R.drawable.ic_search, new SearchFragment());
        addTab("Notifications", R.drawable.ic_notifications, new NotificationsFragment());
        addTab("Messages", R.drawable.ic_message, new MessagesFragment());
        addTab("Profile", R.drawable.ic_person, new ProfileFragment());
    }

    public void addTab(String title, int iconResId, Fragment fragment) {
        tabInfoList.add(new TabInfo(title, iconResId, fragment));
        notifyItemInserted(tabInfoList.size() - 1);
    }

    public void removeTab(int position) {
        if (position >= 0 && position < tabInfoList.size()) {
            tabInfoList.remove(position);
            notifyItemRemoved(position);
        }
    }

    public TabInfo getTabInfo(int position) {
        if (position >= 0 && position < tabInfoList.size()) {
            return tabInfoList.get(position);
        }
        return null;
    }

    public void updateBadgeCount(int position, int count) {
        TabInfo tabInfo = getTabInfo(position);
        if (tabInfo != null) {
            tabInfo.badgeCount = count;
        }
    }

    @NonNull
    @Override
    public Fragment createFragment(int position) {
        TabInfo tabInfo = getTabInfo(position);
        return tabInfo != null ? tabInfo.fragment : new HomeFragment();
    }

    @Override
    public int getItemCount() {
        return tabInfoList.size();
    }
}
```

## ViewPager2 with TabLayout

### Enhanced Tab Integration
```java
public class EnhancedTabsActivity extends AppCompatActivity {

    private TabLayout tabLayout;
    private ViewPager2 viewPager;
    private AdvancedTabsPagerAdapter pagerAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_enhanced_tabs);

        setupToolbar();
        setupTabs();
        setupTabCallbacks();
    }

    private void setupTabs() {
        tabLayout = findViewById(R.id.tabLayout);
        viewPager = findViewById(R.id.viewPager);

        pagerAdapter = new AdvancedTabsPagerAdapter(this);
        viewPager.setAdapter(pagerAdapter);

        new TabLayoutMediator(tabLayout, viewPager,
                (tab, position) -> {
                    AdvancedTabsPagerAdapter.TabInfo tabInfo = pagerAdapter.getTabInfo(position);
                    if (tabInfo != null) {
                        tab.setText(tabInfo.title);
                        tab.setIcon(tabInfo.iconResId);
                        
                        // Set custom view for badges
                        if (tabInfo.badgeCount > 0) {
                            tab.setCustomView(createTabWithBadge(tabInfo));
                        }
                    }
                }).attach();
    }

    private View createTabWithBadge(AdvancedTabsPagerAdapter.TabInfo tabInfo) {
        View customView = LayoutInflater.from(this)
                .inflate(R.layout.custom_tab_badge, null);

        ImageView iconView = customView.findViewById(R.id.tab_icon);
        TextView titleView = customView.findViewById(R.id.tab_title);
        TextView badgeView = customView.findViewById(R.id.tab_badge);

        iconView.setImageResource(tabInfo.iconResId);
        titleView.setText(tabInfo.title);

        if (tabInfo.badgeCount > 0) {
            badgeView.setVisibility(View.VISIBLE);
            badgeView.setText(String.valueOf(tabInfo.badgeCount));
        } else {
            badgeView.setVisibility(View.GONE);
        }

        return customView;
    }

    private void setupTabCallbacks() {
        // Handle tab selection
        tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                int position = tab.getPosition();
                
                // Handle tab selection logic
                onTabSelected(position);
                
                // Animate tab selection
                animateTabSelection(tab);
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {
                // Handle tab unselection
                animateTabUnselection(tab);
            }

            @Override
            public void onTabReselected(TabLayout.Tab tab) {
                // Handle tab reselection (scroll to top, refresh, etc.)
                int position = tab.getPosition();
                handleTabReselection(position);
            }
        });

        // Handle page changes
        viewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                super.onPageSelected(position);
                
                // Update UI based on selected page
                updateUIForPage(position);
            }

            @Override
            public void onPageScrolled(int position, float positionOffset, 
                                     int positionOffsetPixels) {
                super.onPageScrolled(position, positionOffset, positionOffsetPixels);
                
                // Handle scroll animations if needed
            }
        });
    }

    private void onTabSelected(int position) {
        // Implement tab-specific logic
        switch (position) {
            case 0: // Home
                // Setup home tab
                break;
            case 1: // Search
                // Setup search tab
                break;
            case 2: // Notifications
                // Clear notification badge
                clearNotificationBadge();
                break;
        }
    }

    private void animateTabSelection(TabLayout.Tab tab) {
        View customView = tab.getCustomView();
        if (customView != null) {
            customView.animate()
                    .scaleX(1.1f)
                    .scaleY(1.1f)
                    .setDuration(200)
                    .start();
        }
    }

    private void animateTabUnselection(TabLayout.Tab tab) {
        View customView = tab.getCustomView();
        if (customView != null) {
            customView.animate()
                    .scaleX(1.0f)
                    .scaleY(1.0f)
                    .setDuration(200)
                    .start();
        }
    }

    private void handleTabReselection(int position) {
        // Scroll to top or refresh content
        Fragment fragment = getCurrentFragment(position);
        if (fragment instanceof RefreshableFragment) {
            ((RefreshableFragment) fragment).refresh();
        }
    }

    private Fragment getCurrentFragment(int position) {
        return getSupportFragmentManager()
                .findFragmentByTag("f" + position);
    }

    private void updateUIForPage(int position) {
        // Update action bar, FAB, or other UI elements
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            AdvancedTabsPagerAdapter.TabInfo tabInfo = pagerAdapter.getTabInfo(position);
            if (tabInfo != null) {
                actionBar.setTitle(tabInfo.title);
            }
        }
    }

    private void clearNotificationBadge() {
        pagerAdapter.updateBadgeCount(2, 0);
        // Update tab display
        TabLayout.Tab notificationTab = tabLayout.getTabAt(2);
        if (notificationTab != null) {
            AdvancedTabsPagerAdapter.TabInfo tabInfo = pagerAdapter.getTabInfo(2);
            if (tabInfo != null) {
                notificationTab.setCustomView(createTabWithBadge(tabInfo));
            }
        }
    }

    public interface RefreshableFragment {
        void refresh();
    }
}
```

## Custom Tab Layouts

### Custom Tab View
```xml
<!-- custom_tab_badge.xml -->
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:orientation="vertical"
    android:gravity="center"
    android:padding="8dp">

    <FrameLayout
        android:layout_width="wrap_content"
        android:layout_height="wrap_content">

        <ImageView
            android:id="@+id/tab_icon"
            android:layout_width="24dp"
            android:layout_height="24dp"
            android:layout_gravity="center" />

        <TextView
            android:id="@+id/tab_badge"
            android:layout_width="16dp"
            android:layout_height="16dp"
            android:layout_gravity="top|end"
            android:layout_marginTop="-8dp"
            android:layout_marginEnd="-8dp"
            android:background="@drawable/badge_background"
            android:gravity="center"
            android:textColor="@android:color/white"
            android:textSize="10sp"
            android:visibility="gone" />

    </FrameLayout>

    <TextView
        android:id="@+id/tab_title"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="4dp"
        android:textSize="12sp"
        android:textColor="?android:textColorPrimary" />

</LinearLayout>
```

### Badge Background
```xml
<!-- drawable/badge_background.xml -->
<shape xmlns:android="http://schemas.android.com/apk/res/android"
    android:shape="oval">
    <solid android:color="#FF4444" />
</shape>
```

### Custom Tab Styling
```xml
<!-- styles.xml -->
<resources>
    <style name="CustomTabLayout" parent="Widget.Design.TabLayout">
        <item name="tabBackground">@drawable/tab_selector</item>
        <item name="tabTextAppearance">@style/CustomTabTextAppearance</item>
        <item name="tabSelectedTextColor">@color/tab_selected_color</item>
        <item name="tabTextColor">@color/tab_unselected_color</item>
        <item name="tabIndicatorColor">@color/tab_indicator_color</item>
        <item name="tabIndicatorHeight">3dp</item>
        <item name="tabMode">fixed</item>
        <item name="tabGravity">fill</item>
    </style>

    <style name="CustomTabTextAppearance" parent="TextAppearance.Design.Tab">
        <item name="textAllCaps">false</item>
        <item name="android:textSize">14sp</item>
        <item name="android:fontFamily">@font/custom_font</item>
    </style>
</resources>
```

## Dynamic Tab Management

### Dynamic Tab Addition/Removal
```java
public class DynamicTabsActivity extends AppCompatActivity {

    private TabLayout tabLayout;
    private ViewPager2 viewPager;
    private DynamicTabsPagerAdapter pagerAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dynamic_tabs);

        setupTabs();
        setupDynamicControls();
    }

    private void setupTabs() {
        tabLayout = findViewById(R.id.tabLayout);
        viewPager = findViewById(R.id.viewPager);

        pagerAdapter = new DynamicTabsPagerAdapter(this);
        viewPager.setAdapter(pagerAdapter);

        // Initially add some tabs
        addInitialTabs();

        connectTabLayoutWithViewPager();
    }

    private void addInitialTabs() {
        pagerAdapter.addTab("Home", R.drawable.ic_home, new HomeFragment());
        pagerAdapter.addTab("Search", R.drawable.ic_search, new SearchFragment());
    }

    private void connectTabLayoutWithViewPager() {
        new TabLayoutMediator(tabLayout, viewPager,
                (tab, position) -> {
                    DynamicTabsPagerAdapter.TabInfo tabInfo = pagerAdapter.getTabInfo(position);
                    if (tabInfo != null) {
                        tab.setText(tabInfo.title);
                        tab.setIcon(tabInfo.iconResId);
                    }
                }).attach();
    }

    private void setupDynamicControls() {
        Button addTabButton = findViewById(R.id.addTabButton);
        Button removeTabButton = findViewById(R.id.removeTabButton);

        addTabButton.setOnClickListener(v -> showAddTabDialog());
        removeTabButton.setOnClickListener(v -> showRemoveTabDialog());

        // Long press to remove tab
        setupTabLongClickListener();
    }

    private void showAddTabDialog() {
        String[] tabOptions = {"Favorites", "Notifications", "Messages", "Profile", "Settings"};

        new AlertDialog.Builder(this)
                .setTitle("Add Tab")
                .setItems(tabOptions, (dialog, which) -> {
                    String selectedTab = tabOptions[which];
                    addTabByName(selectedTab);
                })
                .show();
    }

    private void addTabByName(String tabName) {
        switch (tabName) {
            case "Favorites":
                pagerAdapter.addTab("Favorites", R.drawable.ic_favorite, new FavoritesFragment());
                break;
            case "Notifications":
                pagerAdapter.addTab("Notifications", R.drawable.ic_notifications, new NotificationsFragment());
                break;
            case "Messages":
                pagerAdapter.addTab("Messages", R.drawable.ic_message, new MessagesFragment());
                break;
            case "Profile":
                pagerAdapter.addTab("Profile", R.drawable.ic_person, new ProfileFragment());
                break;
            case "Settings":
                pagerAdapter.addTab("Settings", R.drawable.ic_settings, new SettingsFragment());
                break;
        }

        // Scroll to newly added tab
        Handler handler = new Handler(Looper.getMainLooper());
        handler.postDelayed(() -> {
            tabLayout.getTabAt(pagerAdapter.getItemCount() - 1).select();
        }, 100);
    }

    private void showRemoveTabDialog() {
        if (pagerAdapter.getItemCount() <= 1) {
            Toast.makeText(this, "Cannot remove the last tab", Toast.LENGTH_SHORT).show();
            return;
        }

        List<String> tabTitles = new ArrayList<>();
        for (int i = 0; i < pagerAdapter.getItemCount(); i++) {
            DynamicTabsPagerAdapter.TabInfo tabInfo = pagerAdapter.getTabInfo(i);
            if (tabInfo != null) {
                tabTitles.add(tabInfo.title);
            }
        }

        String[] tabArray = tabTitles.toArray(new String[0]);

        new AlertDialog.Builder(this)
                .setTitle("Remove Tab")
                .setItems(tabArray, (dialog, which) -> {
                    pagerAdapter.removeTab(which);
                })
                .show();
    }

    private void setupTabLongClickListener() {
        // This requires custom TabLayout implementation for long click detection
        // Or use OnTabSelectedListener with timing logic
    }

    public static class DynamicTabsPagerAdapter extends FragmentStateAdapter {

        private final List<TabInfo> tabInfoList;

        public static class TabInfo {
            public String title;
            public int iconResId;
            public Fragment fragment;

            public TabInfo(String title, int iconResId, Fragment fragment) {
                this.title = title;
                this.iconResId = iconResId;
                this.fragment = fragment;
            }
        }

        public DynamicTabsPagerAdapter(@NonNull FragmentActivity fragmentActivity) {
            super(fragmentActivity);
            this.tabInfoList = new ArrayList<>();
        }

        public void addTab(String title, int iconResId, Fragment fragment) {
            tabInfoList.add(new TabInfo(title, iconResId, fragment));
            notifyItemInserted(tabInfoList.size() - 1);
        }

        public void removeTab(int position) {
            if (position >= 0 && position < tabInfoList.size()) {
                tabInfoList.remove(position);
                notifyItemRemoved(position);
            }
        }

        public TabInfo getTabInfo(int position) {
            if (position >= 0 && position < tabInfoList.size()) {
                return tabInfoList.get(position);
            }
            return null;
        }

        @NonNull
        @Override
        public Fragment createFragment(int position) {
            TabInfo tabInfo = getTabInfo(position);
            return tabInfo != null ? tabInfo.fragment : new Fragment();
        }

        @Override
        public int getItemCount() {
            return tabInfoList.size();
        }

        @Override
        public long getItemId(int position) {
            return tabInfoList.get(position).hashCode();
        }

        @Override
        public boolean containsItem(long itemId) {
            for (TabInfo tabInfo : tabInfoList) {
                if (tabInfo.hashCode() == itemId) {
                    return true;
                }
            }
            return false;
        }
    }
}
```

## Scrollable Tabs

### Scrollable Tab Configuration
```xml
<!-- Scrollable TabLayout -->
<com.google.android.material.tabs.TabLayout
    android:id="@+id/tabLayout"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:background="?attr/colorPrimary"
    app:tabMode="scrollable"
    app:tabGravity="start"
    app:tabTextColor="@android:color/white"
    app:tabSelectedTextColor="@android:color/white"
    app:tabIndicatorColor="@android:color/white"
    app:tabMinWidth="72dp"
    app:tabMaxWidth="264dp" />
```

### Scrollable Tabs Implementation
```java
public class ScrollableTabsActivity extends AppCompatActivity {

    private TabLayout tabLayout;
    private ViewPager2 viewPager;
    private List<CategoryInfo> categories;

    public static class CategoryInfo {
        public String name;
        public int iconResId;
        public Fragment fragment;

        public CategoryInfo(String name, int iconResId, Fragment fragment) {
            this.name = name;
            this.iconResId = iconResId;
            this.fragment = fragment;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scrollable_tabs);

        loadCategories();
        setupTabs();
    }

    private void loadCategories() {
        categories = new ArrayList<>();
        
        // Add many categories to demonstrate scrolling
        categories.add(new CategoryInfo("Technology", R.drawable.ic_computer, new TechnologyFragment()));
        categories.add(new CategoryInfo("Sports", R.drawable.ic_sports, new SportsFragment()));
        categories.add(new CategoryInfo("Entertainment", R.drawable.ic_movie, new EntertainmentFragment()));
        categories.add(new CategoryInfo("Business", R.drawable.ic_business, new BusinessFragment()));
        categories.add(new CategoryInfo("Health", R.drawable.ic_health, new HealthFragment()));
        categories.add(new CategoryInfo("Science", R.drawable.ic_science, new ScienceFragment()));
        categories.add(new CategoryInfo("Travel", R.drawable.ic_travel, new TravelFragment()));
        categories.add(new CategoryInfo("Food", R.drawable.ic_food, new FoodFragment()));
        categories.add(new CategoryInfo("Fashion", R.drawable.ic_fashion, new FashionFragment()));
        categories.add(new CategoryInfo("Music", R.drawable.ic_music, new MusicFragment()));
    }

    private void setupTabs() {
        tabLayout = findViewById(R.id.tabLayout);
        viewPager = findViewById(R.id.viewPager);

        ScrollableTabsPagerAdapter adapter = new ScrollableTabsPagerAdapter(this, categories);
        viewPager.setAdapter(adapter);

        new TabLayoutMediator(tabLayout, viewPager,
                (tab, position) -> {
                    CategoryInfo category = categories.get(position);
                    tab.setText(category.name);
                    tab.setIcon(category.iconResId);
                }).attach();

        // Set initial tab selection
        TabLayout.Tab defaultTab = tabLayout.getTabAt(0);
        if (defaultTab != null) {
            defaultTab.select();
        }
    }

    private static class ScrollableTabsPagerAdapter extends FragmentStateAdapter {

        private final List<CategoryInfo> categories;

        public ScrollableTabsPagerAdapter(@NonNull FragmentActivity fragmentActivity,
                                        List<CategoryInfo> categories) {
            super(fragmentActivity);
            this.categories = categories;
        }

        @NonNull
        @Override
        public Fragment createFragment(int position) {
            return categories.get(position).fragment;
        }

        @Override
        public int getItemCount() {
            return categories.size();
        }
    }
}
```

## Best Practices

### Tab Design Guidelines
- **Limit tab count**: Use 3-5 tabs for fixed tabs, more for scrollable
- **Clear labels**: Use descriptive, concise tab titles
- **Consistent icons**: Use consistent icon style and size
- **Appropriate content**: Each tab should contain substantial content
- **Logical grouping**: Group related content together

### Performance Optimization
- **Lazy loading**: Load tab content when needed
- **Fragment lifecycle**: Properly manage fragment lifecycle
- **Memory management**: Release resources when tabs are not visible
- **Smooth transitions**: Optimize animations and transitions

### Accessibility
- **Content descriptions**: Provide descriptions for tab icons
- **Focus management**: Handle focus properly for keyboard navigation
- **Screen readers**: Ensure tabs work with screen readers
- **Color contrast**: Maintain proper contrast ratios

### User Experience
- **Swipe gestures**: Enable swipe navigation between tabs
- **Tab reselection**: Handle tab reselection appropriately
- **State preservation**: Preserve tab state across orientation changes
- **Visual feedback**: Provide clear visual feedback for selected tabs

Understanding tab navigation and ViewPager enables creating organized, user-friendly interfaces that allow efficient content browsing and navigation in Android applications.
