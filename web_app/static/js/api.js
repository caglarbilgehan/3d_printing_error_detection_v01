/**
 * API JavaScript
 * Handles all API calls to backend
 */

// Dashboard API
async function updateDashboardStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const statusElem = document.getElementById('print-status');
        const motionElem = document.getElementById('motion-ratio');
        const uptimeElem = document.getElementById('uptime');
        const framesElem = document.getElementById('frame-count');
        
        if (statusElem) {
            statusElem.textContent = data.is_printing ? 'PRINTING' : 'IDLE';
            statusElem.style.color = data.is_printing ? '#28a745' : '#6c757d';
        }
        
        if (motionElem) motionElem.textContent = data.motion_ratio.toFixed(2) + '%';
        if (uptimeElem) uptimeElem.textContent = data.uptime + 's';
        if (framesElem) framesElem.textContent = data.frame_count;
    } catch (error) {
        console.error('Dashboard status error:', error);
    }
}

// OctoPrint API
function updateOctoPrintConnection() {
    fetch('/api/octoprint/connection')
        .then(r => r.json())
        .then(data => {
            const statusElem = document.getElementById('connection-status');
            if (statusElem) {
                const status = data.current?.state || 'Disconnected';
                const color = status === 'Operational' ? 'success' : 'danger';
                statusElem.innerHTML = `
                    <span class="status-badge bg-${color}">${status}</span>
                    <p class="mt-2">Port: ${data.current?.port || 'N/A'}</p>
                    <p>Baudrate: ${data.current?.baudrate || 'N/A'}</p>
                `;
            }
        })
        .catch(err => console.error('OctoPrint connection error:', err));
}

function updateOctoPrintPrinter() {
    fetch('/api/octoprint/printer')
        .then(r => r.json())
        .then(data => {
            if (data.temperature) {
                const tool = data.temperature.tool0 || {};
                const bed = data.temperature.bed || {};
                
                const nozzleTempElem = document.getElementById('nozzle-temp');
                const nozzleTargetElem = document.getElementById('nozzle-target');
                const bedTempElem = document.getElementById('bed-temp');
                const bedTargetElem = document.getElementById('bed-target');
                
                if (nozzleTempElem) nozzleTempElem.textContent = `${Math.round(tool.actual || 0)}°C`;
                if (nozzleTargetElem) nozzleTargetElem.textContent = Math.round(tool.target || 0);
                if (bedTempElem) bedTempElem.textContent = `${Math.round(bed.actual || 0)}°C`;
                if (bedTargetElem) bedTargetElem.textContent = Math.round(bed.target || 0);
            }
        })
        .catch(err => console.error('OctoPrint printer error:', err));
}

function updateOctoPrintJob() {
    fetch('/api/octoprint/job')
        .then(r => r.json())
        .then(data => {
            const jobInfoElem = document.getElementById('job-info');
            const jobProgressElem = document.getElementById('job-progress');
            
            if (data.job && data.job.file.name) {
                const progress = Math.round(data.progress.completion || 0);
                const timeLeft = formatTime(data.progress.printTimeLeft);
                
                if (jobInfoElem) {
                    jobInfoElem.innerHTML = `
                        <p><strong>File:</strong> ${data.job.file.name}</p>
                        <p><strong>Time Left:</strong> ${timeLeft}</p>
                        <p><strong>State:</strong> ${data.state}</p>
                    `;
                }
                
                if (jobProgressElem) {
                    jobProgressElem.style.width = `${progress}%`;
                    jobProgressElem.textContent = `${progress}%`;
                }
            } else {
                if (jobInfoElem) jobInfoElem.innerHTML = '<p>No active job</p>';
                if (jobProgressElem) {
                    jobProgressElem.style.width = '0%';
                    jobProgressElem.textContent = '0%';
                }
            }
        })
        .catch(err => console.error('OctoPrint job error:', err));
}

function updateOctoPrintDashboard() {
    fetch('/api/octoprint/dashboard')
        .then(r => r.json())
        .then(data => {
            // Temperature
            if (data.printer && data.printer.temperature) {
                const tool = data.printer.temperature.tool0 || {};
                const bed = data.printer.temperature.bed || {};
                
                const octoNozzleTemp = document.getElementById('octo-nozzle-temp');
                const octoNozzleTarget = document.getElementById('octo-nozzle-target');
                const octoBedTemp = document.getElementById('octo-bed-temp');
                const octoBedTarget = document.getElementById('octo-bed-target');
                
                if (octoNozzleTemp) octoNozzleTemp.textContent = Math.round(tool.actual || 0);
                if (octoNozzleTarget) octoNozzleTarget.textContent = Math.round(tool.target || 0);
                if (octoBedTemp) octoBedTemp.textContent = Math.round(bed.actual || 0);
                if (octoBedTarget) octoBedTarget.textContent = Math.round(bed.target || 0);
            }
            
            // Job
            if (data.job && data.job.job && data.job.job.file.name) {
                const progress = Math.round(data.job.progress.completion || 0);
                const timeLeft = formatTime(data.job.progress.printTimeLeft);
                
                const octoJobName = document.getElementById('octo-job-name');
                const octoJobProgress = document.getElementById('octo-job-progress');
                const octoTimeLeft = document.getElementById('octo-time-left');
                
                if (octoJobName) octoJobName.textContent = data.job.job.file.name;
                if (octoJobProgress) {
                    octoJobProgress.style.width = progress + '%';
                    octoJobProgress.textContent = progress + '%';
                }
                if (octoTimeLeft) octoTimeLeft.textContent = timeLeft;
            } else {
                const octoJobName = document.getElementById('octo-job-name');
                const octoJobProgress = document.getElementById('octo-job-progress');
                const octoTimeLeft = document.getElementById('octo-time-left');
                
                if (octoJobName) octoJobName.textContent = 'No active job';
                if (octoJobProgress) {
                    octoJobProgress.style.width = '0%';
                    octoJobProgress.textContent = '0%';
                }
                if (octoTimeLeft) octoTimeLeft.textContent = '--';
            }
        })
        .catch(err => console.error('OctoPrint dashboard error:', err));
}

// OctoPrint Control Functions
function setNozzleTemp() {
    const temp = document.getElementById('nozzle-input').value;
    fetch('/api/octoprint/printer/temperature', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tool: 'tool0', target: parseInt(temp)})
    }).then(() => alert('Temperature set!'));
}

function setBedTemp() {
    const temp = document.getElementById('bed-input').value;
    fetch('/api/octoprint/printer/bed', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({target: parseInt(temp)})
    }).then(() => alert('Bed temperature set!'));
}

function startJob() {
    fetch('/api/octoprint/job/start', {method: 'POST'})
        .then(() => alert('Print started!'));
}

function pauseJob() {
    fetch('/api/octoprint/job/pause', {method: 'POST'})
        .then(() => alert('Print paused!'));
}

function resumeJob() {
    fetch('/api/octoprint/job/resume', {method: 'POST'})
        .then(() => alert('Print resumed!'));
}

function cancelJob() {
    if (confirm('Are you sure you want to cancel the print?')) {
        fetch('/api/octoprint/job/cancel', {method: 'POST'})
            .then(() => alert('Print cancelled!'));
    }
}

function homeAll() {
    fetch('/api/octoprint/printer/home', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({axes: ['x', 'y', 'z']})
    }).then(() => alert('Homing all axes!'));
}

function homeAxis(axis) {
    fetch('/api/octoprint/printer/home', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({axes: [axis]})
    }).then(() => alert(`Homing ${axis.toUpperCase()} axis!`));
}

function sendGcode() {
    const command = document.getElementById('gcode-input').value;
    if (command) {
        fetch('/api/octoprint/printer/command', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({command: command})
        }).then(() => {
            const output = document.getElementById('gcode-output');
            if (output) {
                output.innerHTML += `<div>> ${command}</div>`;
            }
            document.getElementById('gcode-input').value = '';
        });
    }
}

// ROI API
function saveROI() {
    const points = window.roiPoints || [];
    if (points.length < 3) {
        alert('Minimum 3 points required!');
        return;
    }
    
    fetch('/api/roi', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({points: points})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            const successMsg = document.getElementById('successMsg');
            if (successMsg) {
                successMsg.style.display = 'block';
                setTimeout(() => {
                    successMsg.style.display = 'none';
                }, 3000);
            }
        }
    });
}

function clearROI() {
    if (confirm('Clear all points?')) {
        window.roiPoints = [];
        updatePointList();
        
        fetch('/api/roi', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({reset: true})
        });
    }
}
