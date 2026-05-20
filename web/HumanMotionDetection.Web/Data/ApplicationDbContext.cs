using HumanMotionDetection.Web.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Diagnostics;

namespace HumanMotionDetection.Web.Data
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
        }

        public DbSet<ActivityLog> ActivityLogs { get; set; }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.ConfigureWarnings(w => w.Ignore(RelationalEventId.PendingModelChangesWarning));
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);
            
            modelBuilder.Entity<ActivityLog>().HasData(
                new ActivityLog { Id = 1, CameraId = "CAM-101", BehaviorType = "Normal", ConfidenceScore = 0.98f, Latitude = 21.0285, Longitude = 105.8542, ImagePath = "/images/sample_ok.jpg", CreatedDate = DateTime.UtcNow.AddHours(-2) },
                new ActivityLog { Id = 2, CameraId = "CAM-102", BehaviorType = "Intruding", ConfidenceScore = 0.85f, Latitude = 21.0305, Longitude = 105.8522, ImagePath = "/images/sample_alert.jpg", CreatedDate = DateTime.UtcNow.AddHours(-1) }
            );
        }
    }
}
