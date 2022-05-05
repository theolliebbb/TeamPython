using System;
using Microsoft.EntityFrameworkCore;
using System.Diagnostics.CodeAnalysis;

namespace Dogs.Models
{
    public class DogContext : DbContext

    {
        public DogContext(DbContextOptions<DogContext> options)
            : base(options)
        {
        }
        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseInMemoryDatabase("Dogs");
        }

        public DbSet<Dog> Dogs { get; set; } = null!;
    }
}
