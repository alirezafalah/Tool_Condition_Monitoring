# Tool Profile Visualizer - UI Improvements

## Overview
Complete UI modernization with matplotlib integration and professional custom widgets.

## New Features

### 1. Matplotlib Plot Widget (matplotlib_plot_widget.py)
- **Purpose**: Replaced SVG viewer with direct CSV plotting for accurate coordinate mapping
- **Features**:
  - Direct CSV data loading and plotting
  - Accurate degree indicator using matplotlib's `axvline()`
  - Support for both raw and processed CSV files
  - Proper x-axis (degree) to coordinate mapping
  - Interactive plot with zoom/pan capabilities

### 2. Custom UI Widgets (custom_widgets.py)

#### ToggleSwitch
- iOS-style animated toggle switch
- Smooth animation (200ms cubic easing)
- Green/gray color scheme for on/off states
- Emits `toggled(bool)` signal on state change

#### CircleCheckBox
- Professional circular checkbox design
- Gradient fill when checked (green)
- Hover effects for better UX
- Consistent styling with the toggle switch

#### ToggleSwitchWithLabels
- Composite widget combining toggle with labels
- Dynamic label styling (bold white for active, gray for inactive)
- Perfect for "Raw/Processed" graph type selection
- Forwards signals from internal toggle switch

### 3. Enhanced Profile Window (profile_window.py)

#### Plot Controls
- **Graph Type Toggle**: Switch between Raw and Processed data
  - Uses custom ToggleSwitchWithLabels
  - Reloads appropriate CSV file
  - Updates plot and degree indicator automatically

- **Degree Indicator**: Shows current viewing position on plot
  - Uses CircleCheckBox for toggle
  - Red vertical line on matplotlib plot
  - Accurate coordinate mapping from CSV data

#### Image Navigation
- **Image Counter**: Shows "Image X/Total" (e.g., "Image 1/360")
  - Updates dynamically with navigation
  - Styled in blue with bold font
  - 1-indexed for user-friendly display

- **Auto-load First Image**: 
  - Automatically displays first degree (0.00°) on startup
  - No need to manually enter a degree
  - Immediate visual feedback

#### ROI Visualization
- **ROI Line on Mask**: Toggle red line showing ROI boundary
  - Uses CircleCheckBox for consistent UI
  - Draws horizontal line at configured ROI height
  - Metadata-driven (reads roi_height from JSON)

## Technical Improvements

### Coordinate Accuracy
- **Problem**: SVG coordinate mapping was inaccurate for degree indicator
- **Solution**: Matplotlib plot directly from CSV where degree values are explicit
- **Result**: Perfect alignment between degree value and plot position

### Code Organization
- Separated matplotlib widget into dedicated file
- Custom widgets in reusable module
- Clean signal forwarding from composite widgets
- Proper metadata loading from JSON

### Data Flow
1. CSV paths determined from tool_id
2. Metadata loaded for ROI height
3. Raw CSV loaded on startup
4. Toggle switches between raw/processed CSV
5. Degree indicator updates on image navigation
6. Image counter updates with current position

## Files Modified/Created

### New Files
- `src/matplotlib_plot_widget.py` - Matplotlib integration
- `src/custom_widgets.py` - Professional UI controls

### Modified Files
- `src/profile_window.py` - Complete UI overhaul
  - Removed ZoomableSvgWidget dependency
  - Added MatplotlibPlotWidget
  - Replaced QRadioButton/QCheckBox with custom widgets
  - Added image counter
  - Implemented auto-load on startup

## Configuration

### Metadata Requirements
The viewer expects metadata JSON files in:
```
DATA/1d_profiles/analysis_metadata/{tool_id}_metadata.json
```

Required metadata structure:
```json
{
  "roi_parameters": {
    "roi_height": 200
  },
  "analysis_date": "2025-01-01 12:00:00",
  "total_images_analyzed": 360
}
```

### CSV File Paths
- Raw: `DATA/1d_profiles/{tool_id}_area_vs_angle.csv`
- Processed: `DATA/1d_profiles/{tool_id}_area_vs_angle_processed.csv`

## User Experience Enhancements

1. **Professional Appearance**: iOS-style controls, smooth animations
2. **Clear State Indication**: Bold labels, color changes, visual feedback
3. **Accurate Visualization**: Matplotlib ensures correct degree positioning
4. **Progress Tracking**: Image counter shows position in sequence
5. **Immediate Startup**: First image auto-loads
6. **Flexible Analysis**: Easy switching between raw/processed data
7. **ROI Context**: Visual indication of analysis region on mask

## Future Enhancements (Optional)

- [ ] Add export functionality for current view
- [ ] Implement keyboard shortcuts (e.g., Space for toggle)
- [ ] Add zoom controls for matplotlib plot
- [ ] Save view preferences (last viewed degree, toggle states)
- [ ] Multi-tool comparison view
- [ ] Animation mode (auto-advance through degrees)
