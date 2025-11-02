/**
 * Error Detection JavaScript
 * Handles error detection UI updates
 */

// Update Error Detection in Sidebar
function updateSidebarErrors() {
    fetch('/api/errors')
        .then(r => r.json())
        .then(data => {
            // Total errors
            const totalElem = document.getElementById('sidebar-total-errors');
            if (totalElem) {
                totalElem.textContent = data.total_errors || 0;
            }
            
            // Baseline status
            const baselineStatus = document.getElementById('sidebar-baseline-status');
            if (baselineStatus) {
                if (data.baseline_established) {
                    baselineStatus.innerHTML = '<i class="bi bi-check-circle"></i> Baseline oluşturuldu';
                } else {
                    baselineStatus.innerHTML = '<i class="bi bi-hourglass-split"></i> Baseline oluşturuluyor...';
                }
            }
            
            // Update each error
            const errorTypes = ['separation', 'underextrusion', 'deformation', 'surface_defect', 'model_deviation'];
            errorTypes.forEach(errorType => {
                const errorData = data.details?.[errorType];
                if (errorData) {
                    const item = document.getElementById(`sidebar-error-${errorType}`);
                    const badge = document.getElementById(`sidebar-badge-${errorType}`);
                    
                    if (item && badge) {
                        if (errorData.detected) {
                            item.classList.add('detected');
                            badge.className = 'badge bg-danger badge-xs';
                            badge.textContent = 'ERR';
                        } else {
                            item.classList.remove('detected');
                            badge.className = 'badge bg-success badge-xs';
                            badge.textContent = 'OK';
                        }
                    }
                }
            });
        })
        .catch(err => console.error('Error detection error:', err));
}

// Update Error Cards (Detailed view)
function updateErrorCards() {
    fetch('/api/errors')
        .then(r => r.json())
        .then(data => {
            // Total errors
            const totalElem = document.getElementById('total-errors');
            if (totalElem) {
                totalElem.textContent = data.total_errors || 0;
            }
            
            // Baseline status
            const baselineElem = document.getElementById('baseline-status');
            if (baselineElem) {
                if (data.baseline_established) {
                    baselineElem.textContent = '✓ Baseline oluşturuldu';
                } else {
                    baselineElem.textContent = 'Baseline oluşturuluyor...';
                }
            }
            
            // Update each error card
            const errorTypes = ['separation', 'underextrusion', 'deformation', 'surface_defect', 'model_deviation'];
            errorTypes.forEach(errorType => {
                const errorData = data.details?.[errorType];
                if (errorData) {
                    const card = document.getElementById(`error-${errorType}`);
                    const badge = document.getElementById(`badge-${errorType}`);
                    const conf = document.getElementById(`conf-${errorType}`);
                    
                    if (card && badge && conf) {
                        if (errorData.detected) {
                            card.classList.add('detected');
                            badge.className = 'badge bg-danger';
                            badge.textContent = 'HATA TESPİT EDİLDİ!';
                            conf.textContent = `${Math.round(errorData.confidence * 100)}%`;
                        } else {
                            card.classList.remove('detected');
                            badge.className = 'badge bg-success';
                            badge.textContent = 'Normal';
                            conf.textContent = '0%';
                        }
                    }
                }
            });
        })
        .catch(err => console.error('Error detection error:', err));
}

// Start error detection updates
setInterval(updateSidebarErrors, 2000);
updateSidebarErrors();
