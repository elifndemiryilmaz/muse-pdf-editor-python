# 📋 PDF Edit API - Deployment Guide

## 🏗️ Architecture Overview

This API implements a **dual-backend architecture**:
- **PHP Backend** (runs on RDS): Handles API requests, file management, PDF generation via MPDF
- **Python Service** (runs locally): Handles PDF element extraction via PyMuPDF

```
Frontend → RDS (PHP) → Local Python Service → PyMuPDF → JSON → PHP → Frontend
```

## 🚀 Deployment Steps

### **1. RDS (PHP Backend) Setup**

#### Upload Files to RDS:
```bash
# Upload the enhanced files
- v1/lib/Controllers/PdfEdit.controller.php
- v1/lib/Models/PdfEdit.model.php  
- v1/router.rules.php
- config/environment.example.php
```

#### Configure Environment:
```bash
# Copy and configure environment file
cp config/environment.example.php config/environment.php

# Edit config/environment.php and set:
PY_SERVICE_URL=http://YOUR_LOCAL_IP:5001
UPLOAD_BASE_DIR=/tmp/intern-uploads
RDS_DEPLOYMENT=true
```

#### Set Environment Variables:
```bash
# In RDS environment, set these variables:
export PY_SERVICE_URL="http://YOUR_LOCAL_IP:5001"
export UPLOAD_BASE_DIR="/tmp/intern-uploads"
export RDS_DEPLOYMENT="true"
export DEBUG_MODE="false"
```

### **2. Local Python Service Setup**

#### Install Dependencies:
```bash
cd python-service
pip install -r requirements.txt
```

#### Start Python Service:
```bash
# Start on port 5001 (accessible from RDS)
python app.py

# Or specify port explicitly:
PORT=5001 python app.py
```

#### Configure Network Access:
```bash
# Make sure your local machine is accessible from RDS
# You may need to:
# 1. Configure firewall to allow port 5001
# 2. Use your public IP address
# 3. Consider using ngrok for testing: ngrok http 5001
```

### **3. Database Setup (if needed)**

#### Create PDF Edit Sessions Table:
```sql
CREATE TABLE pdf_edits (
    id INT AUTO_INCREMENT PRIMARY KEY,
    upload_id VARCHAR(255) UNIQUE NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_path TEXT,
    elements_json LONGTEXT,
    modifications_json LONGTEXT,
    status ENUM('ready', 'processing', 'modified', 'completed', 'error') DEFAULT 'ready',
    output_path TEXT,
    download_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_upload_id (upload_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);
```

## 🔧 Configuration Options

### **Environment Variables**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PY_SERVICE_URL` | Python service endpoint | `http://127.0.0.1:5001` | **YES** for RDS |
| `UPLOAD_BASE_DIR` | Upload directory path | `/tmp/intern-uploads` | No |
| `RDS_DEPLOYMENT` | Enable RDS mode | `false` | No |
| `DEBUG_MODE` | Enable debug logging | `true` | No |
| `MAX_UPLOAD_SIZE` | Max file size (bytes) | `10485760` (10MB) | No |

### **Network Configuration**

For RDS to reach your local Python service:

1. **Use Public IP**: Replace `127.0.0.1` with your public IP
2. **Port Forwarding**: Configure router/firewall for port 5001  
3. **Tunneling Service**: Use ngrok or similar for testing
4. **VPN**: Set up VPN connection between RDS and local network

## 📡 API Endpoints

### **Enhanced Workflow Endpoints**

```bash
# Start editing session
POST /pdfedit/session/start
Content-Type: multipart/form-data
Body: file=your.pdf

# Get extracted elements
GET /pdfedit/session/{sessionId}/elements

# Save modifications
PUT /pdfedit/session/{sessionId}/modifications
Content-Type: application/json
Body: {"modifications": [...]}

# Generate final PDF
POST /pdfedit/session/{sessionId}/generate
Content-Type: application/json  
Body: {"modifications": [...]}

# Download result
GET /pdfedit/session/{sessionId}/download
```

### **Legacy Endpoints (still supported)**

```bash
# Health check
GET /pdfedit/health

# Convert PDF to HTML/elements
POST /pdfedit/convert
Content-Type: multipart/form-data
Body: file=your.pdf

# Generate PDF from HTML (disabled - use new workflow)
POST /pdfedit/generate
```

## 🧪 Testing the Deployment

### **1. Test Connectivity**
```bash
# From RDS, test Python service
curl http://YOUR_LOCAL_IP:5001/health

# Expected response:
{"status": "ok", "service": "python"}
```

### **2. Test PDF Upload**
```bash
# Upload PDF and start session
curl -X POST \
  https://YOUR_RDS_DOMAIN/pdfedit/session/start \
  -F "file=@test.pdf"

# Expected response:
{
  "message": "success",
  "content": {
    "sessionId": "abc123...",
    "elements": [...],
    "html": "...",
    "originalFileName": "test.pdf"
  }
}
```

### **3. Test Full Workflow**
```bash
# 1. Start session (upload PDF)
SESSION_ID=$(curl -X POST ... | jq -r '.content.sessionId')

# 2. Get elements
curl https://YOUR_RDS_DOMAIN/pdfedit/session/$SESSION_ID/elements

# 3. Save modifications
curl -X PUT \
  https://YOUR_RDS_DOMAIN/pdfedit/session/$SESSION_ID/modifications \
  -H "Content-Type: application/json" \
  -d '{"modifications": [{"text": "New text", "x": 100, "y": 200, "fontSize": 14}]}'

# 4. Generate final PDF
curl -X POST \
  https://YOUR_RDS_DOMAIN/pdfedit/session/$SESSION_ID/generate \
  -H "Content-Type: application/json" \
  -d '{"modifications": [...]}'

# 5. Download result
curl https://YOUR_RDS_DOMAIN/pdfedit/session/$SESSION_ID/download
```

## 🚨 Troubleshooting

### **Common Issues**

1. **"Python service unreachable"**
   - Check if Python service is running: `curl http://localhost:5001/health`
   - Verify network connectivity from RDS to local machine
   - Check firewall settings on local machine

2. **"Failed to create upload directory"**
   - Verify RDS has write permissions to upload directory
   - Check `UPLOAD_BASE_DIR` environment variable
   - Try using `/tmp/intern-uploads` as fallback

3. **"Upload error"**
   - Check PHP upload limits: `upload_max_filesize`, `post_max_size`
   - Verify file permissions on upload directory
   - Check disk space on RDS

4. **"PDF generation failed"**
   - Ensure MPDF dependencies are installed
   - Check PHP memory limits
   - Verify FPDI can read source PDF files

### **Debug Mode**

Enable debug mode for detailed error messages:
```php
// In config/environment.php
$_SERVER['DEBUG_MODE'] = true;
```

### **Logs**

Check logs for errors:
```bash
# PHP logs (RDS)
tail -f /var/log/apache2/error.log

# Python service logs  
python app.py  # Will show console output
```

## 🔒 Security Considerations

1. **File Upload Security**
   - Only PDF files are allowed
   - File size limits enforced
   - Upload directory isolation

2. **Network Security** 
   - Use HTTPS for production
   - Consider VPN for Python service access
   - Implement rate limiting

3. **Data Privacy**
   - Temporary files are cleaned up
   - Session data expires after use
   - No sensitive data in logs (production mode)

## 📈 Performance Optimization

1. **File Management**
   - Implement cleanup cron job for old files
   - Use separate storage for large files
   - Cache frequently accessed elements

2. **Network Optimization**
   - Use compression for API responses
   - Implement request timeouts
   - Consider CDN for downloads

3. **Database Optimization**
   - Add indexes for session queries
   - Implement session expiration
   - Use connection pooling

## 🎯 Next Steps

1. **Monitor deployment** with health checks
2. **Set up automated cleanup** for temporary files  
3. **Implement logging and monitoring**
4. **Add authentication** if needed
5. **Scale Python service** if traffic increases

---

**Need help?** Check the troubleshooting section or review the code comments in the enhanced controllers.
