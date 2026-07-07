"""
================================================================
demo.py — Quick Demonstration for Supervisor
================================================================
"""
import subprocess
import sys

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║     GENERATIVE DESIGN OF HSS BEAMS USING REINFORCEMENT LEARNING              ║
║     LIVE DEMONSTRATION                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

This demonstration shows how our trained AI agent designs
optimal HSS beams for real construction projects.
""")

# Demo 1: Standard office building beam
print("\n" + "="*80)
print("DEMO 1: Typical Office Building Floor Beam")
print("="*80)
print("Scenario: 10m span, 60 kN/m load, 20-storey building")
input("Press Enter to run AI design...")

subprocess.run([
    "python", "design_beam.py",
    "--model", "models/hss_exp54/best_model.zip",
    "--span", "10",
    "--load", "60",
    "--storey", "20",
    "--verbose"
])

# Demo 2: Heavy industrial beam
print("\n" + "="*80)
print("DEMO 2: Heavy Industrial Loading")
print("="*80)
print("Scenario: 15m span, 120 kN/m heavy machinery load")
input("Press Enter to run AI design...")

subprocess.run([
    "python", "design_beam.py",
    "--model", "models/hss_exp54/best_model.zip",
    "--span", "15",
    "--load", "120",
    "--storey", "5",
    "--optimize", "mass"
])

# Demo 3: Optimization comparison
print("\n" + "="*80)
print("DEMO 3: Multi-Objective Optimization")
print("="*80)
print("Same beam, different optimization goals")
print("Comparing: Minimum Mass vs Minimum Cost vs Balanced")
input("Press Enter to compare...")

for optimize in ['mass', 'cost', 'balanced']:
    print(f"\n--- Optimization: {optimize.upper()} ---")
    subprocess.run([
        "python", "design_beam.py",
        "--model", "models/hss_exp54/best_model.zip",
        "--span", "12",
        "--load", "60",
        "--optimize", optimize
    ])

print("\n" + "="*80)
print("DEMONSTRATION COMPLETE")
print("="*80)
print("""
Key Achievements:
✓ AI designs beams in <1 second vs hours of manual calculation
✓ Explores thousands of possibilities automatically
✓ Finds optimal solutions considering mass, cost, and CO2
✓ Compliant with Eurocode 3 standards
✓ Handles complex buckling and deflection checks
""")