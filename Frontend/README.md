# MRI Insight Suite Frontend

A modern, responsive React-based web application for advanced MRI analysis and visualization. This frontend provides an intuitive interface for brain tumor analysis, featuring comprehensive medical imaging tools, patient management, and real-time AI-powered diagnostics.

## Overview

A sophisticated medical imaging platform that combines cutting-edge web technologies with advanced medical visualization capabilities, serving as the user interface for the Neuroscan AI system with seamless AI-powered brain tumor analysis tools.

### Key Features

- **Advanced Medical Imaging**: Interactive MRI viewer with multi-planar reconstruction
- **AI-Powered Analysis**: Real-time tumor detection, classification, and segmentation
- **Comprehensive Dashboard**: Statistics, recent scans, and volume analysis charts
- **Patient Management**: Complete scan library with detailed metadata
- **Secure Authentication**: JWT-based user authentication and authorization
- **Responsive Design**: Mobile-first design with dark/light theme support
- **Real-time Updates**: Live processing status and progress tracking

## Technology Stack

### Core Framework

- **React 18.3.1**: Modern React with hooks and concurrent features
- **TypeScript 5.8.3**: Type-safe development with full IntelliSense support
- **Vite 7.3.1**: Fast development server and optimized builds
- **React Router 6.30.1**: Client-side routing with protected routes

### UI Framework & Styling

- **Tailwind CSS 3.4.17**: Utility-first CSS framework with custom design system
- **shadcn/ui**: High-quality, accessible component library
- **Radix UI**: Unstyled, accessible components for custom implementations
- **Framer Motion 12.33.0**: Smooth animations and transitions
- **Lucide React 0.462.0**: Beautiful, consistent icon system

### Medical Imaging & Visualization

- **Cornerstone.js 4.15.31**: Professional medical image viewer
- **Cornerstone NIfTI Loader**: Specialized loader for medical imaging formats
- **VTK.js 34.16.3**: Advanced 3D visualization and processing
- **Recharts 2.15.4**: Interactive charts for data visualization

### State Management & Data Fetching

- **TanStack Query 5.83.0**: Powerful server state management
- **React Hook Form 7.61.1**: Performant forms with validation
- **Zod 3.25.76**: TypeScript-first schema validation

### Development Tools

- **ESLint 9.32.0**: Code quality and consistency
- **Vitest 3.2.4**: Fast unit testing framework
- **Testing Library**: Comprehensive React testing utilities
- **PostCSS**: CSS processing and optimization

## Application Architecture

### Component Architecture

The application follows a modular component architecture with clear separation of concerns:

- **Pages**: Route-level components representing application screens
- **Layout Components**: Navigation, sidebar, and structural elements
- **Feature Components**: Business logic components (scans, dashboard, etc.)
- **UI Components**: Reusable, design system components

### State Management

- **Client State**: React hooks and context for UI state
- **Server State**: TanStack Query for API data management
- **Form State**: React Hook Form with Zod validation
- **Authentication**: Custom context with JWT token management

## Core Features

### Medical Imaging Viewer

Advanced MRI visualization with professional medical imaging capabilities:

- **Multi-planar Views**: Axial, sagittal, and coronal planes
- **Window/Level Adjustment**: Interactive contrast and brightness controls
- **Zoom and Pan**: Smooth navigation within medical images
- **Measurement Tools**: Distance, angle, and area measurements
- **Segmentation Overlay**: AI-generated tumor segmentation visualization
- **NIfTI Support**: Direct loading of medical imaging formats

### AI Analysis Integration

Seamless integration with backend AI models:

- **Real-time Processing**: Live status updates during analysis
- **Multi-modal Support**: FLAIR, T1, T1CE, T2 MRI sequences
- **Results Visualization**: Interactive display of detection, classification, and segmentation results
- **Confidence Scores**: Transparent AI model confidence metrics
- **Volume Analysis**: 3D tumor volume calculations and tracking

### Dashboard Analytics

Comprehensive data visualization and insights:

- **Statistics Cards**: Key metrics and KPIs
- **Volume Charts**: Tumor volume trends over time
- **Recent Scans**: Quick access to latest analyses
- **Processing Status**: Real-time queue and progress monitoring
- **User Statistics**: Personal usage and activity metrics

## User Interface Design

### Design System

- **Typography**: Plus Jakarta Sans for optimal readability
- **Color Palette**: Medical-themed color scheme with accessibility focus
- **Dark Mode**: Complete dark/light theme support with system preference detection
- **Responsive**: Mobile-first design with breakpoint optimization
- **Accessibility**: WCAG 2.1 AA compliance with semantic HTML

### Component Library

Extensive collection of reusable components:

- **Form Components**: Input, select, checkbox, radio groups
- **Navigation**: Sidebar, menubar, breadcrumbs
- **Data Display**: Tables, cards, charts, badges
- **Feedback**: Toasts, alerts, progress indicators
- **Layout**: Grid, flexbox, container utilities

## API Integration

### Authentication Flow

- **JWT Tokens**: Secure token-based authentication
- **Protected Routes**: Automatic redirect for unauthenticated users
- **Token Refresh**: Automatic token renewal
- **User Context**: Global authentication state management

### Data Fetching

- **TanStack Query**: Caching, background updates, and optimistic updates
- **Error Handling**: Comprehensive error boundaries and user feedback
- **Loading States**: Skeleton screens and progress indicators
- **Retry Logic**: Automatic retry with exponential backoff

### File Upload

- **Progress Tracking**: Real-time upload progress
- **Drag & Drop**: Modern file upload interface
- **Validation**: Client-side file format and size validation
- **Multi-file Support**: Batch upload of MRI modalities
