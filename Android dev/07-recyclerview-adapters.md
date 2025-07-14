# RecyclerView and Adapters

## Table of Contents
- [Introduction to RecyclerView](#introduction-to-recyclerview)
- [Setting up RecyclerView](#setting-up-recyclerview)
- [Creating Adapters](#creating-adapters)
- [ViewHolder Pattern](#viewholder-pattern)
- [Layout Managers](#layout-managers)
- [Item Decorations](#item-decorations)
- [Item Interactions](#item-interactions)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)

## Introduction to RecyclerView

**RecyclerView** is a more advanced and flexible version of ListView. It's designed to display large datasets efficiently by recycling views that are no longer visible.

### Key Benefits
- **Memory Efficient**: Recycles views automatically
- **Flexible**: Supports different layout patterns
- **Customizable**: Easy to customize appearance and behavior
- **Performance**: Better scrolling performance
- **Animation Support**: Built-in item animations

### RecyclerView vs ListView
| RecyclerView | ListView |
|--------------|----------|
| ViewHolder pattern enforced | ViewHolder optional |
| Multiple layout managers | Single layout pattern |
| Built-in animations | No built-in animations |
| More flexible | Simple but limited |
| Better performance | Adequate for simple lists |

## Setting up RecyclerView

### 1. Add Dependencies
```gradle
// app/build.gradle
dependencies {
    implementation 'androidx.recyclerview:recyclerview:1.3.1'
}
```

### 2. Add to Layout
```xml
<!-- res/layout/activity_main.xml -->
<androidx.recyclerview.widget.RecyclerView
    android:id="@+id/recyclerView"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="8dp"
    android:clipToPadding="false" />
```

### 3. Basic Setup in Activity
```java
public class MainActivity extends AppCompatActivity {
    private RecyclerView recyclerView;
    private UserAdapter adapter;
    private List<User> userList;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initRecyclerView();
        loadData();
    }
    
    private void initRecyclerView() {
        recyclerView = findViewById(R.id.recyclerView);
        
        // Set layout manager
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        
        // Initialize data
        userList = new ArrayList<>();
        
        // Create and set adapter
        adapter = new UserAdapter(userList, this);
        recyclerView.setAdapter(adapter);
        
        // Optional: Improve performance
        recyclerView.setHasFixedSize(true);
    }
    
    private void loadData() {
        // Add sample data
        userList.add(new User("John Doe", "john@example.com", R.drawable.avatar1));
        userList.add(new User("Jane Smith", "jane@example.com", R.drawable.avatar2));
        userList.add(new User("Bob Johnson", "bob@example.com", R.drawable.avatar3));
        
        // Notify adapter
        adapter.notifyDataSetChanged();
    }
}
```

## Creating Adapters

### 1. Data Model
```java
public class User {
    private String name;
    private String email;
    private int avatarResource;
    
    public User(String name, String email, int avatarResource) {
        this.name = name;
        this.email = email;
        this.avatarResource = avatarResource;
    }
    
    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    
    public int getAvatarResource() { return avatarResource; }
    public void setAvatarResource(int avatarResource) { this.avatarResource = avatarResource; }
}
```

### 2. Item Layout
```xml
<!-- res/layout/item_user.xml -->
<?xml version="1.0" encoding="utf-8"?>
<androidx.cardview.widget.CardView xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:layout_margin="8dp"
    app:cardCornerRadius="8dp"
    app:cardElevation="4dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="horizontal"
        android:padding="16dp"
        android:gravity="center_vertical">

        <ImageView
            android:id="@+id/imageAvatar"
            android:layout_width="60dp"
            android:layout_height="60dp"
            android:src="@drawable/ic_person"
            android:scaleType="centerCrop"
            android:background="@drawable/circle_background" />

        <LinearLayout
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:layout_marginStart="16dp"
            android:orientation="vertical">

            <TextView
                android:id="@+id/textName"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:text="Name"
                android:textSize="18sp"
                android:textStyle="bold"
                android:textColor="@color/black" />

            <TextView
                android:id="@+id/textEmail"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:text="email@example.com"
                android:textSize="14sp"
                android:textColor="@color/gray"
                android:layout_marginTop="4dp" />

        </LinearLayout>

        <ImageView
            android:id="@+id/imageMore"
            android:layout_width="24dp"
            android:layout_height="24dp"
            android:src="@drawable/ic_more_vert"
            android:background="?attr/selectableItemBackgroundBorderless"
            android:padding="4dp" />

    </LinearLayout>

</androidx.cardview.widget.CardView>
```

### 3. Basic Adapter
```java
public class UserAdapter extends RecyclerView.Adapter<UserAdapter.UserViewHolder> {
    
    private List<User> userList;
    private Context context;
    private OnItemClickListener listener;
    
    // Interface for click handling
    public interface OnItemClickListener {
        void onItemClick(User user, int position);
        void onMoreClick(User user, int position);
    }
    
    public UserAdapter(List<User> userList, Context context) {
        this.userList = userList;
        this.context = context;
    }
    
    public void setOnItemClickListener(OnItemClickListener listener) {
        this.listener = listener;
    }
    
    @NonNull
    @Override
    public UserViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_user, parent, false);
        return new UserViewHolder(view);
    }
    
    @Override
    public void onBindViewHolder(@NonNull UserViewHolder holder, int position) {
        User user = userList.get(position);
        holder.bind(user, position);
    }
    
    @Override
    public int getItemCount() {
        return userList.size();
    }
    
    // ViewHolder class
    public class UserViewHolder extends RecyclerView.ViewHolder {
        private ImageView imageAvatar;
        private TextView textName;
        private TextView textEmail;
        private ImageView imageMore;
        
        public UserViewHolder(@NonNull View itemView) {
            super(itemView);
            
            imageAvatar = itemView.findViewById(R.id.imageAvatar);
            textName = itemView.findViewById(R.id.textName);
            textEmail = itemView.findViewById(R.id.textEmail);
            imageMore = itemView.findViewById(R.id.imageMore);
            
            // Set click listeners
            itemView.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (position != RecyclerView.NO_POSITION && listener != null) {
                    listener.onItemClick(userList.get(position), position);
                }
            });
            
            imageMore.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (position != RecyclerView.NO_POSITION && listener != null) {
                    listener.onMoreClick(userList.get(position), position);
                }
            });
        }
        
        public void bind(User user, int position) {
            textName.setText(user.getName());
            textEmail.setText(user.getEmail());
            imageAvatar.setImageResource(user.getAvatarResource());
            
            // Optional: Set alternate background colors
            if (position % 2 == 0) {
                itemView.setBackgroundColor(ContextCompat.getColor(context, R.color.lightGray));
            } else {
                itemView.setBackgroundColor(ContextCompat.getColor(context, android.R.color.white));
            }
        }
    }
    
    // Helper methods for data manipulation
    public void addItem(User user) {
        userList.add(user);
        notifyItemInserted(userList.size() - 1);
    }
    
    public void removeItem(int position) {
        if (position >= 0 && position < userList.size()) {
            userList.remove(position);
            notifyItemRemoved(position);
        }
    }
    
    public void updateItem(int position, User user) {
        if (position >= 0 && position < userList.size()) {
            userList.set(position, user);
            notifyItemChanged(position);
        }
    }
    
    public void updateList(List<User> newList) {
        userList.clear();
        userList.addAll(newList);
        notifyDataSetChanged();
    }
}
```

### 4. Using the Adapter
```java
// In Activity
private void setupAdapter() {
    adapter = new UserAdapter(userList, this);
    adapter.setOnItemClickListener(new UserAdapter.OnItemClickListener() {
        @Override
        public void onItemClick(User user, int position) {
            Toast.makeText(MainActivity.this, "Clicked: " + user.getName(), 
                Toast.LENGTH_SHORT).show();
            // Navigate to detail screen
            Intent intent = new Intent(MainActivity.this, UserDetailActivity.class);
            intent.putExtra("user_name", user.getName());
            startActivity(intent);
        }
        
        @Override
        public void onMoreClick(User user, int position) {
            showContextMenu(user, position);
        }
    });
    
    recyclerView.setAdapter(adapter);
}

private void showContextMenu(User user, int position) {
    PopupMenu popup = new PopupMenu(this, findViewById(R.id.imageMore));
    popup.getMenuInflater().inflate(R.menu.user_context_menu, popup.getMenu());
    
    popup.setOnMenuItemClickListener(item -> {
        switch (item.getItemId()) {
            case R.id.menu_edit:
                editUser(user, position);
                return true;
            case R.id.menu_delete:
                deleteUser(position);
                return true;
            default:
                return false;
        }
    });
    
    popup.show();
}
```

## ViewHolder Pattern

### Manual ViewHolder (Pre-RecyclerView)
```java
// Old ListView approach (don't use this)
public View getView(int position, View convertView, ViewGroup parent) {
    ViewHolder holder;
    
    if (convertView == null) {
        convertView = inflater.inflate(R.layout.item_user, parent, false);
        holder = new ViewHolder();
        holder.textName = convertView.findViewById(R.id.textName);
        holder.textEmail = convertView.findViewById(R.id.textEmail);
        convertView.setTag(holder);
    } else {
        holder = (ViewHolder) convertView.getTag();
    }
    
    // Bind data
    User user = userList.get(position);
    holder.textName.setText(user.getName());
    holder.textEmail.setText(user.getEmail());
    
    return convertView;
}

static class ViewHolder {
    TextView textName;
    TextView textEmail;
}
```

### RecyclerView ViewHolder (Enforced)
```java
public class UserViewHolder extends RecyclerView.ViewHolder {
    // Views are automatically held by ViewHolder
    private TextView textName;
    private TextView textEmail;
    private ImageView imageAvatar;
    
    public UserViewHolder(@NonNull View itemView) {
        super(itemView);
        // Initialize views once
        textName = itemView.findViewById(R.id.textName);
        textEmail = itemView.findViewById(R.id.textEmail);
        imageAvatar = itemView.findViewById(R.id.imageAvatar);
    }
    
    public void bind(User user) {
        // Bind data efficiently
        textName.setText(user.getName());
        textEmail.setText(user.getEmail());
        Glide.with(itemView.getContext())
            .load(user.getAvatarUrl())
            .into(imageAvatar);
    }
}
```

## Layout Managers

### 1. LinearLayoutManager
```java
// Vertical list
LinearLayoutManager layoutManager = new LinearLayoutManager(this);
recyclerView.setLayoutManager(layoutManager);

// Horizontal list
LinearLayoutManager layoutManager = new LinearLayoutManager(this, 
    LinearLayoutManager.HORIZONTAL, false);
recyclerView.setLayoutManager(layoutManager);

// Reverse layout
LinearLayoutManager layoutManager = new LinearLayoutManager(this, 
    LinearLayoutManager.VERTICAL, true);
recyclerView.setLayoutManager(layoutManager);
```

### 2. GridLayoutManager
```java
// Grid with 2 columns
GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 2);
recyclerView.setLayoutManager(gridLayoutManager);

// Grid with different span sizes
GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 3);
gridLayoutManager.setSpanSizeLookup(new GridLayoutManager.SpanSizeLookup() {
    @Override
    public int getSpanSize(int position) {
        // Header items span all columns
        if (adapter.getItemViewType(position) == TYPE_HEADER) {
            return 3;
        }
        // Regular items span 1 column
        return 1;
    }
});
recyclerView.setLayoutManager(gridLayoutManager);
```

### 3. StaggeredGridLayoutManager
```java
// Staggered grid with 2 columns
StaggeredGridLayoutManager staggeredLayoutManager = 
    new StaggeredGridLayoutManager(2, StaggeredGridLayoutManager.VERTICAL);
recyclerView.setLayoutManager(staggeredLayoutManager);

// Handle gap in staggered grid
staggeredLayoutManager.setGapStrategy(StaggeredGridLayoutManager.GAP_HANDLING_MOVE_ITEMS_BETWEEN_SPANS);
```

### 4. Custom Layout Manager
```java
public class CustomLayoutManager extends RecyclerView.LayoutManager {
    
    @Override
    public RecyclerView.LayoutParams generateDefaultLayoutParams() {
        return new RecyclerView.LayoutParams(
            RecyclerView.LayoutParams.WRAP_CONTENT,
            RecyclerView.LayoutParams.WRAP_CONTENT
        );
    }
    
    @Override
    public void onLayoutChildren(RecyclerView.Recycler recycler, RecyclerView.State state) {
        // Custom layout logic
        detachAndScrapAttachedViews(recycler);
        
        // Layout children in custom pattern
        for (int i = 0; i < getItemCount(); i++) {
            View view = recycler.getViewForPosition(i);
            addView(view);
            measureChildWithMargins(view, 0, 0);
            
            // Position view
            int left = i * getDecoratedMeasuredWidth(view) / 2;
            int top = i * 50;
            layoutDecorated(view, left, top, 
                left + getDecoratedMeasuredWidth(view),
                top + getDecoratedMeasuredHeight(view));
        }
    }
    
    @Override
    public boolean canScrollVertically() {
        return true;
    }
    
    @Override
    public int scrollVerticallyBy(int dy, RecyclerView.Recycler recycler, 
                                 RecyclerView.State state) {
        // Handle scrolling
        return dy;
    }
}
```

## Item Decorations

### 1. Simple Divider
```java
public class DividerItemDecoration extends RecyclerView.ItemDecoration {
    
    private Drawable divider;
    
    public DividerItemDecoration(Context context) {
        divider = ContextCompat.getDrawable(context, R.drawable.divider);
    }
    
    @Override
    public void onDraw(@NonNull Canvas c, @NonNull RecyclerView parent, 
                      @NonNull RecyclerView.State state) {
        int left = parent.getPaddingLeft();
        int right = parent.getWidth() - parent.getPaddingRight();
        
        int childCount = parent.getChildCount();
        for (int i = 0; i < childCount; i++) {
            View child = parent.getChildAt(i);
            
            RecyclerView.LayoutParams params = 
                (RecyclerView.LayoutParams) child.getLayoutParams();
            
            int top = child.getBottom() + params.bottomMargin;
            int bottom = top + divider.getIntrinsicHeight();
            
            divider.setBounds(left, top, right, bottom);
            divider.draw(c);
        }
    }
    
    @Override
    public void getItemOffsets(@NonNull Rect outRect, @NonNull View view, 
                              @NonNull RecyclerView parent, @NonNull RecyclerView.State state) {
        outRect.bottom = divider.getIntrinsicHeight();
    }
}

// Usage
recyclerView.addItemDecoration(new DividerItemDecoration(this));
```

### 2. Grid Spacing Decoration
```java
public class GridSpacingItemDecoration extends RecyclerView.ItemDecoration {
    
    private int spanCount;
    private int spacing;
    private boolean includeEdge;
    
    public GridSpacingItemDecoration(int spanCount, int spacing, boolean includeEdge) {
        this.spanCount = spanCount;
        this.spacing = spacing;
        this.includeEdge = includeEdge;
    }
    
    @Override
    public void getItemOffsets(@NonNull Rect outRect, @NonNull View view, 
                              @NonNull RecyclerView parent, @NonNull RecyclerView.State state) {
        int position = parent.getChildAdapterPosition(view);
        int column = position % spanCount;
        
        if (includeEdge) {
            outRect.left = spacing - column * spacing / spanCount;
            outRect.right = (column + 1) * spacing / spanCount;
            
            if (position < spanCount) {
                outRect.top = spacing;
            }
            outRect.bottom = spacing;
        } else {
            outRect.left = column * spacing / spanCount;
            outRect.right = spacing - (column + 1) * spacing / spanCount;
            if (position >= spanCount) {
                outRect.top = spacing;
            }
        }
    }
}

// Usage
int spanCount = 2;
int spacing = 50; // 50px
boolean includeEdge = true;
recyclerView.addItemDecoration(new GridSpacingItemDecoration(spanCount, spacing, includeEdge));
```

### 3. Section Header Decoration
```java
public class SectionHeaderDecoration extends RecyclerView.ItemDecoration {
    
    private Paint paint;
    private Paint textPaint;
    private int headerHeight;
    
    public SectionHeaderDecoration(Context context) {
        headerHeight = (int) (context.getResources().getDisplayMetrics().density * 48);
        
        paint = new Paint();
        paint.setColor(ContextCompat.getColor(context, R.color.headerBackground));
        
        textPaint = new Paint();
        textPaint.setColor(ContextCompat.getColor(context, R.color.headerText));
        textPaint.setTextSize(context.getResources().getDimensionPixelSize(R.dimen.header_text_size));
        textPaint.setAntiAlias(true);
    }
    
    @Override
    public void onDrawOver(@NonNull Canvas c, @NonNull RecyclerView parent, 
                          @NonNull RecyclerView.State state) {
        // Draw floating header
        View topChild = parent.getChildAt(0);
        if (topChild != null) {
            int position = parent.getChildAdapterPosition(topChild);
            String headerText = getHeaderText(position);
            
            c.drawRect(0, 0, parent.getWidth(), headerHeight, paint);
            c.drawText(headerText, 16, headerHeight / 2f + textPaint.getTextSize() / 2f, textPaint);
        }
    }
    
    @Override
    public void getItemOffsets(@NonNull Rect outRect, @NonNull View view, 
                              @NonNull RecyclerView parent, @NonNull RecyclerView.State state) {
        int position = parent.getChildAdapterPosition(view);
        if (isFirstInSection(position)) {
            outRect.top = headerHeight;
        }
    }
    
    private String getHeaderText(int position) {
        // Return header text based on position
        return "Section " + (position / 10);
    }
    
    private boolean isFirstInSection(int position) {
        // Check if this is the first item in a section
        return position % 10 == 0;
    }
}
```

## Item Interactions

### 1. Click Handling
```java
public class UserAdapter extends RecyclerView.Adapter<UserAdapter.UserViewHolder> {
    
    public interface OnItemClickListener {
        void onItemClick(User user, int position);
        void onItemLongClick(User user, int position);
    }
    
    public class UserViewHolder extends RecyclerView.ViewHolder 
            implements View.OnClickListener, View.OnLongClickListener {
        
        public UserViewHolder(@NonNull View itemView) {
            super(itemView);
            itemView.setOnClickListener(this);
            itemView.setOnLongClickListener(this);
        }
        
        @Override
        public void onClick(View v) {
            int position = getAdapterPosition();
            if (position != RecyclerView.NO_POSITION && listener != null) {
                listener.onItemClick(userList.get(position), position);
            }
        }
        
        @Override
        public boolean onLongClick(View v) {
            int position = getAdapterPosition();
            if (position != RecyclerView.NO_POSITION && listener != null) {
                listener.onItemLongClick(userList.get(position), position);
                return true;
            }
            return false;
        }
    }
}
```

### 2. Swipe to Delete
```java
public class SwipeToDeleteCallback extends ItemTouchHelper.SimpleCallback {
    
    private UserAdapter adapter;
    private Context context;
    
    public SwipeToDeleteCallback(UserAdapter adapter, Context context) {
        super(0, ItemTouchHelper.LEFT | ItemTouchHelper.RIGHT);
        this.adapter = adapter;
        this.context = context;
    }
    
    @Override
    public boolean onMove(@NonNull RecyclerView recyclerView, 
                         @NonNull RecyclerView.ViewHolder viewHolder, 
                         @NonNull RecyclerView.ViewHolder target) {
        return false;
    }
    
    @Override
    public void onSwiped(@NonNull RecyclerView.ViewHolder viewHolder, int direction) {
        int position = viewHolder.getAdapterPosition();
        User deletedUser = adapter.getUserAt(position);
        
        // Remove item
        adapter.removeItem(position);
        
        // Show undo snackbar
        Snackbar snackbar = Snackbar.make(viewHolder.itemView, 
            deletedUser.getName() + " deleted", Snackbar.LENGTH_LONG);
        snackbar.setAction("UNDO", v -> {
            adapter.addItem(position, deletedUser);
        });
        snackbar.show();
    }
    
    @Override
    public void onChildDraw(@NonNull Canvas c, @NonNull RecyclerView recyclerView, 
                           @NonNull RecyclerView.ViewHolder viewHolder, 
                           float dX, float dY, int actionState, boolean isCurrentlyActive) {
        super.onChildDraw(c, recyclerView, viewHolder, dX, dY, actionState, isCurrentlyActive);
        
        // Draw delete background
        View itemView = viewHolder.itemView;
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        
        if (dX > 0) { // Swiping right
            c.drawRect(itemView.getLeft(), itemView.getTop(), 
                itemView.getLeft() + dX, itemView.getBottom(), paint);
        } else { // Swiping left
            c.drawRect(itemView.getRight() + dX, itemView.getTop(), 
                itemView.getRight(), itemView.getBottom(), paint);
        }
    }
}

// Usage
ItemTouchHelper itemTouchHelper = new ItemTouchHelper(new SwipeToDeleteCallback(adapter, this));
itemTouchHelper.attachToRecyclerView(recyclerView);
```

### 3. Drag and Drop
```java
public class DragDropCallback extends ItemTouchHelper.SimpleCallback {
    
    private UserAdapter adapter;
    
    public DragDropCallback(UserAdapter adapter) {
        super(ItemTouchHelper.UP | ItemTouchHelper.DOWN, 0);
        this.adapter = adapter;
    }
    
    @Override
    public boolean onMove(@NonNull RecyclerView recyclerView, 
                         @NonNull RecyclerView.ViewHolder viewHolder, 
                         @NonNull RecyclerView.ViewHolder target) {
        int fromPosition = viewHolder.getAdapterPosition();
        int toPosition = target.getAdapterPosition();
        
        adapter.moveItem(fromPosition, toPosition);
        return true;
    }
    
    @Override
    public void onSwiped(@NonNull RecyclerView.ViewHolder viewHolder, int direction) {
        // Not used for drag and drop
    }
    
    @Override
    public void onSelectedChanged(@Nullable RecyclerView.ViewHolder viewHolder, int actionState) {
        super.onSelectedChanged(viewHolder, actionState);
        
        if (actionState == ItemTouchHelper.ACTION_STATE_DRAG) {
            viewHolder.itemView.setAlpha(0.8f);
            viewHolder.itemView.setScaleX(1.05f);
            viewHolder.itemView.setScaleY(1.05f);
        }
    }
    
    @Override
    public void clearView(@NonNull RecyclerView recyclerView, 
                         @NonNull RecyclerView.ViewHolder viewHolder) {
        super.clearView(recyclerView, viewHolder);
        
        viewHolder.itemView.setAlpha(1.0f);
        viewHolder.itemView.setScaleX(1.0f);
        viewHolder.itemView.setScaleY(1.0f);
    }
}

// In adapter
public void moveItem(int fromPosition, int toPosition) {
    User user = userList.remove(fromPosition);
    userList.add(toPosition, user);
    notifyItemMoved(fromPosition, toPosition);
}
```

## Advanced Features

### 1. Multiple View Types
```java
public class MultiTypeAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {
    
    private static final int TYPE_HEADER = 0;
    private static final int TYPE_USER = 1;
    private static final int TYPE_FOOTER = 2;
    
    private List<Object> items;
    
    @Override
    public int getItemViewType(int position) {
        Object item = items.get(position);
        
        if (item instanceof Header) {
            return TYPE_HEADER;
        } else if (item instanceof User) {
            return TYPE_USER;
        } else if (item instanceof Footer) {
            return TYPE_FOOTER;
        }
        
        return TYPE_USER;
    }
    
    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        switch (viewType) {
            case TYPE_HEADER:
                View headerView = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_header, parent, false);
                return new HeaderViewHolder(headerView);
                
            case TYPE_USER:
                View userView = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_user, parent, false);
                return new UserViewHolder(userView);
                
            case TYPE_FOOTER:
                View footerView = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_footer, parent, false);
                return new FooterViewHolder(footerView);
                
            default:
                throw new IllegalArgumentException("Unknown view type: " + viewType);
        }
    }
    
    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        int viewType = getItemViewType(position);
        
        switch (viewType) {
            case TYPE_HEADER:
                ((HeaderViewHolder) holder).bind((Header) items.get(position));
                break;
            case TYPE_USER:
                ((UserViewHolder) holder).bind((User) items.get(position));
                break;
            case TYPE_FOOTER:
                ((FooterViewHolder) holder).bind((Footer) items.get(position));
                break;
        }
    }
    
    // ViewHolder classes
    public class HeaderViewHolder extends RecyclerView.ViewHolder {
        private TextView titleText;
        
        public HeaderViewHolder(@NonNull View itemView) {
            super(itemView);
            titleText = itemView.findViewById(R.id.titleText);
        }
        
        public void bind(Header header) {
            titleText.setText(header.getTitle());
        }
    }
    
    // UserViewHolder and FooterViewHolder...
}
```

### 2. Pagination
```java
public class PaginationScrollListener extends RecyclerView.OnScrollListener {
    
    private LinearLayoutManager layoutManager;
    private OnLoadMoreListener loadMoreListener;
    private boolean isLoading = false;
    private boolean isLastPage = false;
    
    public PaginationScrollListener(LinearLayoutManager layoutManager) {
        this.layoutManager = layoutManager;
    }
    
    @Override
    public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
        super.onScrolled(recyclerView, dx, dy);
        
        int visibleItemCount = layoutManager.getChildCount();
        int totalItemCount = layoutManager.getItemCount();
        int firstVisibleItemPosition = layoutManager.findFirstVisibleItemPosition();
        
        if (!isLoading && !isLastPage) {
            if ((visibleItemCount + firstVisibleItemPosition) >= totalItemCount
                    && firstVisibleItemPosition >= 0) {
                if (loadMoreListener != null) {
                    loadMoreListener.onLoadMore();
                }
            }
        }
    }
    
    public void setLoading(boolean loading) {
        isLoading = loading;
    }
    
    public void setLastPage(boolean lastPage) {
        isLastPage = lastPage;
    }
    
    public void setOnLoadMoreListener(OnLoadMoreListener listener) {
        this.loadMoreListener = listener;
    }
    
    public interface OnLoadMoreListener {
        void onLoadMore();
    }
}

// Usage
PaginationScrollListener scrollListener = new PaginationScrollListener(layoutManager);
scrollListener.setOnLoadMoreListener(() -> {
    // Load more data
    loadNextPage();
});
recyclerView.addOnScrollListener(scrollListener);
```

### 3. Search and Filter
```java
public class SearchableAdapter extends RecyclerView.Adapter<SearchableAdapter.ViewHolder> 
        implements Filterable {
    
    private List<User> originalList;
    private List<User> filteredList;
    
    public SearchableAdapter(List<User> userList) {
        this.originalList = new ArrayList<>(userList);
        this.filteredList = new ArrayList<>(userList);
    }
    
    @Override
    public Filter getFilter() {
        return new Filter() {
            @Override
            protected FilterResults performFiltering(CharSequence constraint) {
                List<User> filteredResults = new ArrayList<>();
                
                if (constraint == null || constraint.length() == 0) {
                    filteredResults.addAll(originalList);
                } else {
                    String filterPattern = constraint.toString().toLowerCase().trim();
                    
                    for (User user : originalList) {
                        if (user.getName().toLowerCase().contains(filterPattern) ||
                            user.getEmail().toLowerCase().contains(filterPattern)) {
                            filteredResults.add(user);
                        }
                    }
                }
                
                FilterResults results = new FilterResults();
                results.values = filteredResults;
                results.count = filteredResults.size();
                return results;
            }
            
            @Override
            protected void publishResults(CharSequence constraint, FilterResults results) {
                filteredList.clear();
                filteredList.addAll((List<User>) results.values);
                notifyDataSetChanged();
            }
        };
    }
    
    @Override
    public int getItemCount() {
        return filteredList.size();
    }
    
    // Other adapter methods use filteredList instead of originalList
}

// Usage with SearchView
@Override
public boolean onCreateOptionsMenu(Menu menu) {
    getMenuInflater().inflate(R.menu.main_menu, menu);
    
    MenuItem searchItem = menu.findItem(R.id.action_search);
    SearchView searchView = (SearchView) searchItem.getActionView();
    
    searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
        @Override
        public boolean onQueryTextSubmit(String query) {
            adapter.getFilter().filter(query);
            return false;
        }
        
        @Override
        public boolean onQueryTextChange(String newText) {
            adapter.getFilter().filter(newText);
            return false;
        }
    });
    
    return true;
}
```

## Best Practices

### 1. Optimize Performance
```java
// Set fixed size if known
recyclerView.setHasFixedSize(true);

// Use appropriate layout manager
recyclerView.setLayoutManager(new LinearLayoutManager(this));

// Prefetch items
if (layoutManager instanceof LinearLayoutManager) {
    ((LinearLayoutManager) layoutManager).setInitialPrefetchItemCount(4);
}

// Use ViewHolder binding efficiently
@Override
public void onBindViewHolder(@NonNull UserViewHolder holder, int position) {
    User user = userList.get(position);
    
    // Only update what's necessary
    if (!TextUtils.equals(holder.textName.getText(), user.getName())) {
        holder.textName.setText(user.getName());
    }
    
    // Use efficient image loading
    Glide.with(holder.itemView.getContext())
        .load(user.getAvatarUrl())
        .placeholder(R.drawable.placeholder)
        .into(holder.imageAvatar);
}
```

### 2. Handle Large Datasets
```java
// Use DiffUtil for efficient updates
public class UserDiffCallback extends DiffUtil.Callback {
    
    private List<User> oldList;
    private List<User> newList;
    
    public UserDiffCallback(List<User> oldList, List<User> newList) {
        this.oldList = oldList;
        this.newList = newList;
    }
    
    @Override
    public int getOldListSize() {
        return oldList.size();
    }
    
    @Override
    public int getNewListSize() {
        return newList.size();
    }
    
    @Override
    public boolean areItemsTheSame(int oldItemPosition, int newItemPosition) {
        return oldList.get(oldItemPosition).getId()
            .equals(newList.get(newItemPosition).getId());
    }
    
    @Override
    public boolean areContentsTheSame(int oldItemPosition, int newItemPosition) {
        User oldUser = oldList.get(oldItemPosition);
        User newUser = newList.get(newItemPosition);
        
        return oldUser.equals(newUser);
    }
}

// Update adapter with DiffUtil
public void updateUsers(List<User> newUsers) {
    UserDiffCallback diffCallback = new UserDiffCallback(userList, newUsers);
    DiffUtil.DiffResult diffResult = DiffUtil.calculateDiff(diffCallback);
    
    userList.clear();
    userList.addAll(newUsers);
    diffResult.dispatchUpdatesTo(this);
}
```

### 3. Memory Management
```java
@Override
public void onViewRecycled(@NonNull UserViewHolder holder) {
    super.onViewRecycled(holder);
    
    // Clear image to prevent memory leaks
    Glide.with(holder.itemView.getContext()).clear(holder.imageAvatar);
    
    // Remove any listeners
    holder.itemView.setOnClickListener(null);
    
    // Cancel any running animations
    holder.itemView.clearAnimation();
}

@Override
public void onDetachedFromRecyclerView(@NonNull RecyclerView recyclerView) {
    super.onDetachedFromRecyclerView(recyclerView);
    
    // Clean up resources
    if (userList != null) {
        userList.clear();
    }
}
```

### 4. Error Handling
```java
@Override
public void onBindViewHolder(@NonNull UserViewHolder holder, int position) {
    try {
        if (position < userList.size()) {
            User user = userList.get(position);
            holder.bind(user);
        }
    } catch (Exception e) {
        Log.e(TAG, "Error binding view at position " + position, e);
        // Handle error gracefully
        holder.showErrorState();
    }
}

// In ViewHolder
public void showErrorState() {
    textName.setText("Error loading data");
    textEmail.setText("");
    imageAvatar.setImageResource(R.drawable.ic_error);
}
```

RecyclerView is a powerful component for displaying lists and grids efficiently. Understanding its architecture, proper adapter implementation, and optimization techniques is crucial for building smooth, responsive Android applications.
