"""
================================================================
design_beam.py
----------------------------------------------------------------
Production Beam Design Script — HSS Beam Generative Design
HKU Research Project

USAGE (Single Design):
    python design_beam.py --model models/run1/best_model.zip --span 8.5 --load 45 --storey 25

USAGE (Batch Design):
    python design_beam.py --model models/run1/best_model.zip --batch batch_input.csv

INPUT FORMAT (batch_input.csv):
    span_m,load_kNm,storey,ltb_factor
    8.5,45,25,0.25
    10.0,60,30,0.25
    12.0,35,20,0.25

================================================================
"""

import os
import csv
import json
import argparse
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env.high_rise_generative_env_claude_final import HighRiseGenerativeEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="HSS Beam Designer — AI-Powered Generative Design"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--span", type=float,
                       help="Beam span in meters")
    parser.add_argument("--load", type=float,
                       help="Design load in kN/m")
    parser.add_argument("--storey", type=int, default=20,
                       help="Number of storeys (default: 20)")
    parser.add_argument("--ltb-factor", type=float, default=0.25,
                       help="Lateral torsional buckling restraint factor")
    parser.add_argument("--sls-factor", type=float, default=0.50,
                       help="Serviceability limit state load factor")
    parser.add_argument("--optimize", choices=['mass','cost','co2','balanced'],
                       default='balanced',
                       help="Optimization objective")
    parser.add_argument("--batch", type=str,
                       help="CSV file for batch processing")
    parser.add_argument("--output", type=str, default="beam_design_results",
                       help="Output file prefix")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed design steps")
    return parser.parse_args()


class BeamDesigner:
    """Production beam designer using trained RL agent"""
    
    def __init__(self, model_path, ltb_factor=0.25, sls_factor=0.50):
        self.model_path = model_path
        self.ltb_factor = ltb_factor
        self.sls_factor = sls_factor
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model and normalization stats"""
        print(f"Loading AI model: {self.model_path}")
        self.model = PPO.load(self.model_path)
        
        # Load VecNormalize if exists
        vecnorm_path = self.model_path.replace("best_model.zip","vecnormalize.pkl")\
                                     .replace("final_model.zip","vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            env = HighRiseGenerativeEnv(
                sls_load_factor=self.sls_factor,
                ltb_restraint_factor=self.ltb_factor
            )
            self.vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env]))
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
            print("  ✓ Normalization stats loaded")
        else:
            self.vecnorm = None
            print("  ⚠ No normalization stats found")
    
    def design_single(self, span_m, load_kNm, storey=20, optimize='balanced',
                     verbose=False):
        """
        Design a single HSS beam
        
        Parameters:
        -----------
        span_m : float
            Beam span in meters
        load_kNm : float
            Design load in kN/m
        storey : int
            Building height in storeys
        optimize : str
            Optimization objective ('mass', 'cost', 'co2', 'balanced')
        verbose : bool
            Show design iteration steps
            
        Returns:
        --------
        dict : Complete beam design with all properties
        """
        
        # Convert to mm and N/mm for environment
        span_mm = span_m * 1000
        load_Nmm = load_kNm  # Environment uses kN/m internally
        
        # Create environment with specific demands
        env = HighRiseGenerativeEnv(
            use_storey_load_scaling=(storey != 20),
            sls_load_factor=self.sls_factor,
            ltb_restraint_factor=self.ltb_factor
        )
        
        # Set the design requirements
        obs, _ = env.reset(seed=42)
        env.span = span_mm
        env.load = load_Nmm
        env.storey = storey
        
        # Generate designs step by step
        designs = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f" DESIGNING HSS BEAM")
            print(f"{'='*80}")
            print(f" Requirements:")
            print(f"   Span: {span_m:.1f} m  |  Load: {load_kNm:.1f} kN/m  |  Storeys: {storey}")
            print(f"   Optimization: {optimize}")
            print(f"\n Design iterations:")
            print(f" {'Step':<6} {'h':>6} {'b':>6} {'tf':>6} {'tw':>6} {'fy':>6} {'Util':>8} {'Mass':>8} {'Status'}")
            print(f" {'-'*70}")
        
        done = False
        step_count = 0
        
        while not done and step_count < 50:  # Max 50 iterations
            # Get model prediction
            if self.vecnorm:
                obs_normalized = self.vecnorm.normalize_obs(env._get_obs())
                action, _ = self.model.predict(obs_normalized, deterministic=True)
            else:
                action, _ = self.model.predict(env._get_obs(), deterministic=True)
            
            # Apply action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # Store design step
            if info.get("ec3"):
                designs.append({
                    'step': step_count,
                    'h': info['h'],
                    'b': info['b'],
                    'tf': info['tf'],
                    'tw': info['tw'],
                    'fy': info['fy'],
                    'utilization': info['utilization'],
                    'mass': info['mass'],
                    'cost': info['cost'],
                    'co2': info['co2'],
                    'section_type': info['section_type'],
                    'section_class': info['ec3'].get('section_class', 0),
                    'feasible': (info['utilization'] <= 1.05 and 
                                info['ec3'].get('section_class', 4) < 4),
                    'moment_util': info['ec3'].get('moment_util', 0),
                    'defl_util': info['ec3'].get('deflection_util', 0)
                })
                
                if verbose:
                    status = "✓ FEASIBLE" if designs[-1]['feasible'] else "✗ FAIL"
                    print(f" {step_count:<6} {info['h']:>6.0f} {info['b']:>6.0f} "
                          f"{info['tf']:>6.1f} {info['tw']:>6.1f} "
                          f"{info['fy']:>6.0f} {info['utilization']:>8.3f} "
                          f"{info['mass']:>8.1f} {status}")
        
        env.close()
        
        # Select the best design
        if not designs:
            return self._empty_result(span_m, load_kNm, storey)
        
        # Filter feasible designs
        feasible = [d for d in designs if d['feasible']]
        
        if feasible:
            # Select based on optimization objective
            if optimize == 'mass':
                best = min(feasible, key=lambda x: x['mass'])
            elif optimize == 'cost':
                best = min(feasible, key=lambda x: x['cost'])
            elif optimize == 'co2':
                best = min(feasible, key=lambda x: x['co2'])
            else:  # balanced - closest to 0.95 utilization
                best = min(feasible, key=lambda x: abs(x['utilization'] - 0.95))
        else:
            # No feasible design, return closest to feasible
            best = min(designs, key=lambda x: abs(x['utilization'] - 0.95))
        
        # Build comprehensive result
        result = {
            'design_id': f"HSS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'input_parameters': {
                'span_m': span_m,
                'load_kNm': load_kNm,
                'storey': storey,
                'optimization': optimize
            },
            'selected_design': {
                'section_height_mm': best['h'],
                'section_width_mm': best['b'],
                'flange_thickness_mm': best['tf'],
                'web_thickness_mm': best['tw'],
                'steel_grade': f"S{int(best['fy'])}",
                'fy_MPa': best['fy'],
                'section_type': best['section_type'],
                'section_class': best['section_class'],
                'mass_per_meter_kg': best['mass'],
                'estimated_cost_gbp': best['cost'],
                'co2_equivalent_kg': best['co2']
            },
            'performance': {
                'utilization_ratio': best['utilization'],
                'moment_utilization': best.get('moment_util', 0),
                'deflection_utilization': best.get('defl_util', 0),
                'is_feasible': best['feasible'],
                'iterations_required': len(designs)
            },
            'alternatives': self._get_alternatives(designs, best, 3),
            'design_path': designs if verbose else []
        }
        
        return result
    
    def _get_alternatives(self, all_designs, selected, n=3):
        """Get alternative design options"""
        feasible = [d for d in all_designs if d['feasible'] and d != selected]
        if not feasible:
            return []
        
        # Sort by different criteria
        alternatives = []
        for d in feasible[:n]:
            alternatives.append({
                'section_height_mm': d['h'],
                'section_width_mm': d['b'],
                'steel_grade': f"S{int(d['fy'])}",
                'mass_kg': d['mass'],
                'utilization': d['utilization'],
                'cost_gbp': d['cost']
            })
        return alternatives[:n]
    
    def _empty_result(self, span_m, load_kNm, storey):
        """Return empty result when design fails"""
        return {
            'design_id': 'FAILED',
            'input_parameters': {
                'span_m': span_m,
                'load_kNm': load_kNm,
                'storey': storey
            },
            'error': 'No valid design found',
            'selected_design': None,
            'performance': {'is_feasible': False}
        }
    
    def design_batch(self, csv_file, optimize='balanced'):
        """Process multiple designs from CSV file"""
        results = []
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            requests = list(reader)
        
        print(f"\n{'='*80}")
        print(f" BATCH BEAM DESIGN — {len(requests)} designs requested")
        print(f"{'='*80}")
        
        for i, req in enumerate(requests, 1):
            span = float(req.get('span_m', req.get('span')))
            load = float(req.get('load_kNm', req.get('load')))
            storey = int(req.get('storey', 20))
            
            print(f"\n[{i}/{len(requests)}] Span: {span}m, Load: {load}kN/m, Storeys: {storey}")
            
            result = self.design_single(span, load, storey, optimize, verbose=True)
            results.append(result)
        
        return results


def format_design_report(result):
    """Create a beautiful design report"""
    if result.get('error'):
        return f"ERROR: {result['error']}"
    
    design = result['selected_design']
    perf = result['performance']
    inp = result['input_parameters']
    
    # Format section designation
    section_name = f"HSS {design['section_height_mm']:.0f}×{design['section_width_mm']:.0f}×{design['flange_thickness_mm']:.1f}×{design['web_thickness_mm']:.1f}"
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HSS BEAM DESIGN REPORT                                     ║
║                    AI-Generated Design {result['design_id']}                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESIGN REQUIREMENTS
────────────────────────────────────────────────────────────────────────────────
  Span Length:      {inp['span_m']:.1f} meters
  Design Load:      {inp['load_kNm']:.1f} kN/m
  Building Height:  {inp['storey']} storeys
  Optimization:     {inp['optimization']}

🏗️  RECOMMENDED SECTION
────────────────────────────────────────────────────────────────────────────────
  Designation:      {section_name}
  Steel Grade:      {design['steel_grade']} (fy = {design['fy_MPa']:.0f} MPa)
  Section Type:     {design['section_type']}
  
  Dimensions:
    • Height (h):   {design['section_height_mm']:.1f} mm
    • Width (b):    {design['section_width_mm']:.1f} mm
    • Flange (tf):  {design['flange_thickness_mm']:.1f} mm
    • Web (tw):     {design['web_thickness_mm']:.1f} mm

📊 STRUCTURAL PERFORMANCE
────────────────────────────────────────────────────────────────────────────────
  Overall Utilization:   {perf['utilization_ratio']:.1%}
  Moment Utilization:    {perf['moment_utilization']:.1%}
  Deflection Check:      {perf['deflection_utilization']:.1%}
  Section Class:         Class {design['section_class']}
  Design Status:         {'✅ FEASIBLE' if perf['is_feasible'] else '❌ NEEDS REVIEW'}
  
  AI Iterations:         {perf['iterations_required']} steps

💰 SUSTAINABILITY & ECONOMICS
────────────────────────────────────────────────────────────────────────────────
  Mass per meter:        {design['mass_per_meter_kg']:.1f} kg/m
  Total beam mass:       {design['mass_per_meter_kg'] * inp['span_m']:.1f} kg
  Estimated Cost:        £{design['estimated_cost_gbp']:.2f}/m
  CO₂ Equivalent:        {design['co2_equivalent_kg']:.1f} kg/m

{ '🔄 ALTERNATIVE OPTIONS' if result.get('alternatives') else ''}
{ '─'*80 if result.get('alternatives') else '' }
"""
    
    if result.get('alternatives'):
        report += f"  {'Option':<10} {'Section':<25} {'Grade':<8} {'Mass':<10} {'Util':<8} {'Cost'}\n"
        report += f"  {'-'*75}\n"
        for i, alt in enumerate(result['alternatives'], 1):
            alt_section = f"HSS {alt['section_height_mm']:.0f}×{alt['section_width_mm']:.0f}"
            report += (f"  Alt {i:<7} {alt_section:<25} {alt['steel_grade']:<8} "
                      f"{alt['mass_kg']:<10.1f} {alt['utilization']:<8.1%} £{alt['cost_gbp']:.2f}\n")
    
    report += f"\n{'═'*80}\n"
    return report


def main():
    args = parse_args()
    
    # Initialize designer
    designer = BeamDesigner(args.model, args.ltb_factor, args.sls_factor)
    
    if args.batch:
        # Batch processing mode
        results = designer.design_batch(args.batch, args.optimize)
        
        # Save batch results
        output_file = f"{args.output}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save CSV summary
        with open(f"{output_file}.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'span_m', 'load_kNm', 'storey', 'feasible', 'utilization',
                'section_height_mm', 'section_width_mm', 'steel_grade',
                'mass_kgm', 'cost_gbp', 'co2_kgm'
            ])
            writer.writeheader()
            for r in results:
                if r.get('selected_design'):
                    writer.writerow({
                        'span_m': r['input_parameters']['span_m'],
                        'load_kNm': r['input_parameters']['load_kNm'],
                        'storey': r['input_parameters']['storey'],
                        'feasible': r['performance']['is_feasible'],
                        'utilization': r['performance']['utilization_ratio'],
                        'section_height_mm': r['selected_design']['section_height_mm'],
                        'section_width_mm': r['selected_design']['section_width_mm'],
                        'steel_grade': r['selected_design']['steel_grade'],
                        'mass_kgm': r['selected_design']['mass_per_meter_kg'],
                        'cost_gbp': r['selected_design']['estimated_cost_gbp'],
                        'co2_kgm': r['selected_design']['co2_equivalent_kg']
                    })
        
        # Save detailed JSON
        with open(f"{output_file}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}.csv and {output_file}.json")
        
        # Print summary table
        print(f"\n{'='*80}")
        print(f" BATCH DESIGN SUMMARY")
        print(f"{'='*80}")
        print(f" {'Span':<8} {'Load':<8} {'Feasible':<10} {'Util':<8} {'Mass':<10} {'Section':<30}")
        print(f" {'-'*80}")
        for r in results:
            if r.get('selected_design'):
                d = r['selected_design']
                p = r['input_parameters']
                print(f" {p['span_m']:<8.1f} {p['load_kNm']:<8.1f} "
                      f"{'✅' if r['performance']['is_feasible'] else '❌':<10} "
                      f"{r['performance']['utilization_ratio']:<8.1%} "
                      f"{d['mass_per_meter_kg']:<10.1f} "
                      f"HSS {d['section_height_mm']:.0f}×{d['section_width_mm']:.0f}×{d['flange_thickness_mm']:.1f}")
    
    else:
        # Single design mode
        if not args.span or not args.load:
            print("ERROR: Either specify --span and --load for single design, or --batch for batch processing")
            return
        
        result = designer.design_single(
            args.span, args.load, args.storey, args.optimize, verbose=args.verbose
        )
        
        # Print beautiful report
        print(format_design_report(result))
        
        # Save to file
        output_file = f"{args.output}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(f"{output_file}.json", 'w') as f:
            # Remove design_path to keep file small
            result_save = {k: v for k, v in result.items() if k != 'design_path'}
            json.dump(result_save, f, indent=2)
        
        print(f"📁 Detailed results saved to: {output_file}.json")
        print(f"   Use --verbose flag to see iteration steps\n")


if __name__ == "__main__":
    main()