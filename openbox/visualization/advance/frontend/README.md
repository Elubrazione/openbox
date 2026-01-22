# Dimensio Visualization Frontend

Interactive visualization dashboard for Dimensio compression history.

## Features

- **Compression Summary**: 4-panel overview showing dimension reduction, compression ratios, range statistics, and text summary
- **Range Compression**: Horizontal bar charts displaying original vs compressed parameter ranges
- **Parameter Importance**: Top-K parameter importance ranking
- **Dimension Evolution**: Line chart showing adaptive dimension changes over iterations
- **Multi-Task Heatmap**: Heatmap visualization of parameter importance across multiple tasks
- **Source Similarities**: Bar chart showing similarity scores between source and target tasks

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Webpack 5** - Module bundler
- **ECharts 5** - Charting library
- **Axios** - HTTP client

## Installation

```bash
npm install
```

## Development

Start the development server:

```bash
npm start
```

The app will open at http://localhost:3000

## Build

Build for production:

```bash
npm run build
```

## API Integration

The frontend is configured to use the real backend API at `http://127.0.0.1:5000/api`.

### Using Real API (Recommended)

1. **Start the Dimensio API server** (in project root):
   ```bash
   python -m api.server
   ```

2. **Start the frontend** (in `front` directory):
   ```bash
   npm start
   ```

3. The frontend will automatically:
   - Connect to the API via webpack proxy
   - Fetch list of experiments
   - Load the first experiment's compression history
   - Display all visualizations

### API Endpoints Used

- `GET /api/experiments` - List all experiments
- `GET /api/experiments/{id}/history` - Get compression history
- `GET /api/experiments/{id}/visualizations` - Get visualization metadata

### Fallback Behavior

If the API is not available, the application will:
1. Display a warning in the console
2. Automatically fall back to mock data
3. Continue to function normally for development/testing

The mock data is based on the sample JSON you provided and includes:
- 12 original parameters → 6 compressed parameters
- 2 pipeline steps (dimension_selection + range_compression)
- Complete compression_info with ranges and ratios

## Chart Components

All chart components are located in `src/components/`:

- `CompressionSummary.tsx` - Main 4-panel summary
- `RangeCompression.tsx` - Range compression visualization
- `ParameterImportance.tsx` - Parameter ranking
- `DimensionEvolution.tsx` - Adaptive dimension tracking
- `MultiTaskHeatmap.tsx` - Multi-task importance heatmap
- `SourceSimilarities.tsx` - Source task similarity bars

## Configuration

- `webpack.config.js` - Webpack configuration
- `tsconfig.json` - TypeScript configuration
- `src/types/index.ts` - Type definitions matching API schemas
