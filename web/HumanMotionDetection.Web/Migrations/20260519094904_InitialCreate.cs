using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

#pragma warning disable CA1814 // Prefer jagged arrays over multidimensional

namespace HumanMotionDetection.Web.Migrations
{
    /// <inheritdoc />
    public partial class InitialCreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "ActivityLogs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    CameraId = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    BehaviorType = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    ConfidenceScore = table.Column<float>(type: "REAL", nullable: false),
                    ImagePath = table.Column<string>(type: "TEXT", nullable: false),
                    Latitude = table.Column<double>(type: "REAL", nullable: false),
                    Longitude = table.Column<double>(type: "REAL", nullable: false),
                    CreatedDate = table.Column<DateTime>(type: "TEXT", nullable: false),
                    IsResolved = table.Column<bool>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ActivityLogs", x => x.Id);
                });

            migrationBuilder.InsertData(
                table: "ActivityLogs",
                columns: new[] { "Id", "BehaviorType", "CameraId", "ConfidenceScore", "CreatedDate", "ImagePath", "IsResolved", "Latitude", "Longitude" },
                values: new object[,]
                {
                    { 1, "Normal", "CAM-101", 0.98f, new DateTime(2026, 5, 19, 7, 49, 3, 887, DateTimeKind.Utc).AddTicks(8672), "/images/sample_ok.jpg", false, 21.028500000000001, 105.85420000000001 },
                    { 2, "Intruding", "CAM-102", 0.85f, new DateTime(2026, 5, 19, 8, 49, 3, 887, DateTimeKind.Utc).AddTicks(9364), "/images/sample_alert.jpg", false, 21.0305, 105.8522 }
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "ActivityLogs");
        }
    }
}
