using System;
using System.ComponentModel.DataAnnotations;

namespace HumanMotionDetection.Web.Models
{
    public class ActivityLog
    {
        public int Id { get; set; }

        [Required]
        [StringLength(50)]
        public string CameraId { get; set; } = string.Empty;

        [Required]
        [StringLength(50)]
        public string BehaviorType { get; set; } = string.Empty;

        public float ConfidenceScore { get; set; }

        [Required]
        public string ImagePath { get; set; } = string.Empty;

        public double Latitude { get; set; }
        public double Longitude { get; set; }

        public DateTime CreatedDate { get; set; } = DateTime.UtcNow;

        public bool IsResolved { get; set; } = false;
    }
}
