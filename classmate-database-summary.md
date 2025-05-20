# ClassMate: Database Design and Implementation

## Overview
ClassMate is an academic organization application designed to help students manage their educational resources, communications, and schedules. It consists of both a mobile application (offline-first approach using Flutter) and a web application (online approach using HTML, CSS, and JavaScript), both utilizing Supabase as the backend service provider.

## 1. Database Architecture Comparison

### Mobile Application Database
- **Type**: Local database with occasional synchronization (offline-first)
- **Structure**: Simplified schema focused on essential features
- **Storage**: Local device storage for files with cloud backup
- **Purpose**: Optimized for offline usage and performance on mobile devices

### Web Application Database
- **Type**: Cloud-based database (online-only)
- **Structure**: More complex schema with user authentication integration
- **Storage**: Cloud storage with direct access to files
- **Purpose**: Designed for consistent access across multiple devices with persistent internet connection

## 2. Mobile Application Database Structure

### Tables and Relationships

#### `chats` Table
| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| id | UUID | Primary key | Unique identifier for chat groups |
| chat_name | Text | Name of the chat group | Identifies the specific chat group |
| admin_id | Text | ID of the chat admin | Controls chat administration |
| university | Text | University name | Groups chats by institution |
| faculty | Text | Faculty name | Groups chats by faculty |
| academic_year | Integer | Academic year | Groups chats by year level |
| created_at | Timestamp | Creation timestamp | Tracks when the chat was created |
| admin_name | Text | Name of admin | Displays admin information |

#### `messages` Table
| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| id | UUID | Primary key | Unique identifier for messages |
| chat_id | UUID | Foreign key to chats table | Links messages to specific chat groups |
| sender_id | Text | ID of message sender | Tracks message ownership |
| sender_name | Text | Name of sender | Displays sender information |
| content | Text | Message content | Stores actual message text |
| created_at | Timestamp | Creation timestamp | Orders messages chronologically |
| is_voice | Boolean | Voice message flag | Indicates if message is voice-based |
| voice_url | Text | URL to voice message | Links to stored voice recording |

#### `summaries` Table
| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| id | UUID | Primary key | Unique identifier for summaries |
| university | Text | University name | Groups summaries by institution |
| timestamp | Timestamp | Creation timestamp | Orders summaries chronologically |
| faculty | Text | Faculty name | Groups summaries by faculty |
| academic_year | Integer | Academic year | Groups summaries by year level |
| file_url | Text | URL to file | Links to stored file |
| file_name | Text | Name of file | Displays file information |
| subject | Text | Subject name | Categorizes summaries by subject |
| uploaded_by | Text | Uploader's identifier | Tracks file ownership |

### Storage Buckets
1. **summaries**: Stores PDF files and educational materials
2. **voice-messages**: Stores voice recordings from the chat feature

### Relationships Diagram (Mobile App)
```
chats (1) ---- (*) messages
     |
     v
filters user content by
university, faculty, academic_year
     |
     v
summaries (filtered by same attributes)
```

## 3. Web Application Database Structure

### Tables and Relationships

#### `students` Table
| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| user_id | UUID | Primary key (from auth.users) | Unique identifier linked to authentication system |
| full_name | Text | Student's full name | Displays user information |
| email | Text | Student's email | Communication and login |
| faculty | Text | Faculty name | Groups students by faculty |
| academic_year | Integer | Academic year | Groups students by year level |
| created_at | Timestamp | Account creation time | Tracks account age |

#### `files` Table
| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| id | UUID | Primary key | Unique identifier for files |
| user_id | UUID | Foreign key to auth.users | Links files to specific users |
| file_name | Text | Name of file | Displays file information |
| file_path | Text | Path to file in storage | Locates the actual file |
| file_type | Text | Type/extension of file | Determines file handling |
| file_size | Integer | Size of file in bytes | Storage management |
| uploaded_at | Timestamp | Upload timestamp | Tracks file age |

#### `schedules` Table
| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| id | UUID | Primary key | Unique identifier for schedule entries |
| user_id | UUID | Foreign key to auth.users | Links schedules to specific users |
| class_name | Text | Name of the class | Identifies specific classes |
| day | Integer | Day of week (1-7) | Determines which day the class occurs |
| start_time | Time | Class start time | Start time of the class |
| end_time | Time | Class end time | End time of the class |
| room | Text | Classroom location | Physical location information |
| professor | Text | Professor's name | Instructor information |
| created_at | Timestamp | Creation timestamp | Tracks when entry was created |

#### `archive` Table
| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| id | Integer | Primary key | Unique identifier for archive entries |
| user_id | UUID | Foreign key to auth.users | Links archives to specific users |
| user_name | Text | Name of uploader | Displays uploader information |
| faculty | Text | Faculty name | Groups archives by faculty |
| academic_year | Integer | Academic year | Groups archives by year level |
| file_name | Text | Name of file | Displays file information |
| file_path | Text | Path to file in storage | Locates the actual file |
| file_type | Text | Type/extension of file | Determines file handling |
| file_size | Integer | Size of file in bytes | Storage management |
| uploaded_at | Timestamp | Upload timestamp | Tracks file age |

#### `tasks` Table
| Field | Type | Description | Purpose |
|-------|------|-------------|---------|
| id | UUID | Primary key | Unique identifier for tasks |
| user_id | UUID | Foreign key to auth.users | Links tasks to specific users |
| title | Text | Task title | Brief description of task |
| description | Text | Task details | Extended information about task |
| due_date | Date | Task deadline | Date when task is due |
| priority | Text | Priority level | Importance classification |
| completed | Boolean | Completion status | Tracks task completion |
| created_at | Timestamp | Creation timestamp | Tracks when task was created |

### Relationships Diagram (Web App)
```
                auth.users (1)
                     |
        +------------+------------+
        |            |            |
        v            v            v
students (1)    files (*)    tasks (*)
        |            
        |            
        v            
schedules (*)      archive (*)
```

## 4. Database Role in Application Lifecycle

### Mobile Application (Offline-First Approach)

1. **Initial Setup**:
   - User installs app and enters personal information (name, university, faculty, academic year)
   - Local database created on device
   - Initial synchronization with Supabase if internet available

2. **Daily Usage**:
   - Data primarily read/written to local device storage
   - Chat messages stored locally and sent to server when connection available
   - Files cached locally for offline access

3. **Synchronization**:
   - Periodic background sync when internet connection detected
   - New messages and files uploaded to cloud
   - New content from other users downloaded

4. **Content Filtering**:
   - Database queries filter content based on university, faculty, and academic year
   - Chat groups and summaries displayed only for relevant cohort

5. **Performance Considerations**:
   - Local database optimized for quick access and minimal storage
   - Media files (voice messages, PDFs) stored in separate storage areas

### Web Application (Online Approach)

1. **User Authentication**:
   - Registration/login using Supabase authentication
   - Email verification required
   - User profile stored in students table

2. **Data Management**:
   - All data stored in Supabase cloud database
   - Real-time synchronization across devices
   - Files uploaded directly to cloud storage

3. **Feature Implementation**:
   - Calendar events stored in schedules table
   - Tasks managed through tasks table with due dates and priorities
   - File sharing through files and archive tables

4. **Content Organization**:
   - Files categorized by type and ownership
   - Archive accessible based on faculty/academic year filters
   - Schedule displayed using calendar interface

5. **Security Implementation**:
   - Authentication through Supabase auth system
   - Row-level security policies for data access
   - User can only modify their own data

## 5. Key Differences Between Mobile and Web Database Implementations

### Data Access Pattern Differences

| Aspect | Mobile Database | Web Database |
|--------|----------------|--------------|
| **Connection Model** | Local-first with occasional sync | Always connected to cloud |
| **Data Storage** | Primarily on device with cloud backup | Primarily in cloud with local caching |
| **Authentication** | Simple local profile, no auth required | Full authentication with email verification |
| **Real-time Updates** | Periodic synchronization | Immediate updates |
| **Offline Capability** | Full functionality without internet | Limited or no offline functionality |

### Schema Differences

| Mobile Database | Web Database | Reason |
|----------------|--------------|--------|
| Simplified chat system with global chats | No built-in chat system | Mobile focuses on communication, web on resource management |
| Limited to essential tables | More comprehensive schema | Web app has expanded functionality |
| No direct authentication integration | Integrated with auth.users | Web requires secure multi-device access |
| Designed for local storage constraints | Designed for cloud scalability | Different deployment environments |

### Storage Approach Differences

| Mobile Storage | Web Storage | Advantage |
|----------------|-------------|-----------|
| Files stored locally with URL references | Files stored in cloud with path references | Mobile: Offline access / Web: Universal access |
| Two storage buckets (summaries, voice-messages) | Integrated file storage system | Mobile: Segregated for performance / Web: Unified for simplicity |
| Voice messages as special type | Generic file types | Mobile: Optimized for communication / Web: Flexibility |

### User Experience Implications

| Mobile Database Design | Web Database Design | User Benefit |
|------------------------|----------------------|-------------|
| Optimized for intermittent connectivity | Requires persistent connection | Mobile: Usable anywhere / Web: Always synchronized |
| Fast local queries | Potentially slower cloud queries | Mobile: Immediate response / Web: Most recent data |
| Limited by device storage | Limited only by cloud storage | Mobile: Portable / Web: Expansive storage |
| Chat-centric social features | Schedule and file management focus | Mobile: Communication / Web: Organization |

## 6. Conclusion: Integration of Both Platforms

The ClassMate application demonstrates a sophisticated approach to database design by tailoring each platform's database structure to match its usage patterns:

1. **Complementary Functionality**: 
   - Mobile app focuses on communication and quick access to resources
   - Web app emphasizes comprehensive resource management and scheduling

2. **Shared Data Model**:
   - Both platforms use similar conceptual models (users, files, academic context)
   - Implementation differs based on platform constraints and strengths

3. **Unified User Experience**:
   - Students can use both platforms interchangeably based on situation
   - Mobile for on-the-go access, web for detailed planning and management

4. **Technological Advantages**:
   - Mobile leverages device capabilities (offline access, notifications)
   - Web leverages browser capabilities (larger displays, advanced filtering)

This dual-platform approach with tailored database designs provides students with a comprehensive solution that adapts to their changing needs throughout the academic lifecycle.