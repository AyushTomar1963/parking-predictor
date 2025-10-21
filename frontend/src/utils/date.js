/**
 * Date utility functions
 * Helper functions for formatting timestamps
 */

/**
 * Format timestamp string to readable local time
 * @param {string} timestamp - ISO timestamp string
 * @returns {string} Formatted date string
 */
export function formatTimestamp(timestamp) {
  try {
    const date = new Date(timestamp)
    
    // Format as: "Dec 31, 2024 23:00"
    const options = {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    }
    
    return date.toLocaleString('en-US', options)
  } catch (error) {
    console.error('Error formatting timestamp:', error)
    return timestamp
  }
}

/**
 * Get hour from timestamp
 * @param {string} timestamp - ISO timestamp string
 * @returns {number} Hour (0-23)
 */
export function getHourFromTimestamp(timestamp) {
  try {
    const date = new Date(timestamp)
    return date.getHours()
  } catch (error) {
    console.error('Error extracting hour:', error)
    return 0
  }
}

/**
 * Format duration in minutes to human-readable string
 * @param {number} minutes - Duration in minutes
 * @returns {string} Formatted duration
 */
export function formatDuration(minutes) {
  if (minutes < 60) {
    return `${minutes.toFixed(0)} min`
  }
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60
  return `${hours}h ${mins.toFixed(0)}min`
}
