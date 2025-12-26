#!/usr/bin/env python3
"""
Monte Carlo Simulation for Light Transmission Through Hair and Scalp (v2)
===========================================================================
Improved model with realistic hair geometry for LED device contact simulation.

Simulates 630nm LED light (2mW/cm²) penetration through hair follicles to scalp tissue.

Key improvements:
- Realistic hair strand geometry with cylindrical model
- Proper hair density calculation for contact zones
- More accurate light-hair interaction modeling

Author: Light Transmission Simulation v2
Date: 2024
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict
import time

np.random.seed(42)


@dataclass
class OpticalProperties:
    """Optical properties of tissue at 630nm wavelength"""
    mu_a: float  # Absorption coefficient (cm^-1)
    mu_s: float  # Scattering coefficient (cm^-1)
    g: float     # Anisotropy factor
    n: float     # Refractive index

    @property
    def mu_t(self) -> float:
        return self.mu_a + self.mu_s

    @property
    def mu_s_prime(self) -> float:
        return self.mu_s * (1 - self.g)

    @property
    def albedo(self) -> float:
        if self.mu_t == 0:
            return 0
        return self.mu_s / self.mu_t


@dataclass
class HairGeometry:
    """Hair strand geometry parameters"""
    diameter: float          # Hair diameter (cm)
    density: float           # Follicles per cm²
    compression_factor: float  # How much hair is compressed when LED is pressed (0-1)


class ImprovedScalpSimulation:
    """
    Improved Monte Carlo simulation for light transmission through hair to scalp.

    Model:
    - LED pressed against scalp surface
    - Hair strands create partial coverage of the illumination area
    - Light can pass through gaps between hairs or through hair strands
    """

    def __init__(self, num_photons: int = 100000,
                 hair_type: str = "dark_brown",
                 include_hair: bool = True,
                 hair_density_factor: float = 1.0):
        """
        Initialize simulation.

        Args:
            num_photons: Number of photons to simulate
            hair_type: Type of hair (black, dark_brown, light_brown, blonde, gray)
            include_hair: Whether to include hair in simulation
            hair_density_factor: Multiplier for hair density (for sensitivity analysis)
        """
        self.num_photons = num_photons
        self.hair_type = hair_type
        self.include_hair = include_hair
        self.hair_density_factor = hair_density_factor

        # LED parameters
        self.wavelength = 630  # nm
        self.irradiance = 2.0  # mW/cm²

        self._init_optical_properties()
        self._init_geometry()

        # Statistics
        self.stats = {
            "total_photons": num_photons,
            "reached_epidermis": 0,
            "reached_dermis": 0,
            "absorbed_in_hair": 0,
            "absorbed_in_epidermis": 0,
            "absorbed_in_dermis": 0,
            "reflected": 0,
            "transmitted_through_hair": 0,
            "passed_through_gap": 0
        }

        # Depth profile for energy deposition
        self.depth_bins = np.linspace(0, 0.5, 101)  # 0-5mm
        self.energy_deposition = np.zeros(100)

    def _init_optical_properties(self):
        """Initialize optical properties for 630nm wavelength"""

        # Hair melanin absorption coefficients (cm^-1) at 630nm
        hair_absorption = {
            "black": 12.0,       # High eumelanin
            "dark_brown": 7.0,   # Moderate eumelanin
            "light_brown": 4.0,
            "blonde": 2.0,       # Low melanin
            "gray": 1.0,         # Very low melanin
            "white": 0.5
        }

        mu_a_hair = hair_absorption.get(self.hair_type, 7.0)

        # Hair shaft optical properties
        # References:
        # - Jacques SL, "Optical properties of biological tissues" (2013)
        # - Bashkatov et al., "Optical properties of skin" (2011)
        self.hair_props = OpticalProperties(
            mu_a=mu_a_hair,
            mu_s=350.0,        # High scattering from keratin
            g=0.85,            # Forward scattering
            n=1.55
        )

        # Epidermis (150μm thick)
        self.epidermis_props = OpticalProperties(
            mu_a=3.0,          # Melanin absorption
            mu_s=420.0,        # High scattering
            g=0.79,
            n=1.4
        )

        # Dermis (2-4mm thick)
        self.dermis_props = OpticalProperties(
            mu_a=0.4,          # Lower absorption (blood)
            mu_s=280.0,        # Collagen scattering
            g=0.90,
            n=1.4
        )

        # Subcutaneous fat
        self.fat_props = OpticalProperties(
            mu_a=0.3,
            mu_s=200.0,
            g=0.9,
            n=1.44
        )

    def _init_geometry(self):
        """Initialize geometric parameters"""

        # Hair geometry
        # Average hair density: 100-150 follicles/cm² on scalp
        # Hair diameter: 50-100 μm
        base_density = 120 * self.hair_density_factor  # follicles/cm²

        self.hair = HairGeometry(
            diameter=0.007,          # 70 μm = 0.007 cm
            density=base_density,
            compression_factor=0.3    # Hair compressed to 30% when LED pressed
        )

        # Calculate hair coverage fraction
        # When LED is pressed, hair strands are compressed and spread
        hair_radius = self.hair.diameter / 2
        hair_cross_section = np.pi * hair_radius ** 2

        # Effective coverage includes hair spreading when compressed
        effective_hair_area = hair_cross_section / self.hair.compression_factor
        self.hair_coverage = min(0.85, self.hair.density * effective_hair_area)

        # Layer thicknesses (cm)
        self.hair_layer_thickness = 0.05   # 500μm compressed hair layer
        self.epidermis_thickness = 0.015   # 150μm
        self.dermis_thickness = 0.25       # 2.5mm

        # Z-coordinates of layer boundaries
        self.z_surface = 0.0
        if self.include_hair:
            self.z_epidermis_start = self.hair_layer_thickness
        else:
            self.z_epidermis_start = 0.0

        self.z_dermis_start = self.z_epidermis_start + self.epidermis_thickness
        self.z_fat_start = self.z_dermis_start + self.dermis_thickness

    def _sample_step_size(self, mu_t: float) -> float:
        """Sample step size from exponential distribution"""
        if mu_t <= 0:
            return 1e6  # Effectively infinite
        return -np.log(np.random.random()) / mu_t

    def _henyey_greenstein_scatter(self, direction: np.ndarray, g: float) -> np.ndarray:
        """Sample new direction using Henyey-Greenstein phase function"""

        if abs(g) < 1e-6:
            cos_theta = 2 * np.random.random() - 1
        else:
            temp = (1 - g**2) / (1 - g + 2*g*np.random.random())
            cos_theta = (1 + g**2 - temp**2) / (2 * g)
            cos_theta = np.clip(cos_theta, -1, 1)

        sin_theta = np.sqrt(max(0, 1 - cos_theta**2))
        phi = 2 * np.pi * np.random.random()

        dx, dy, dz = direction

        if abs(dz) > 0.99999:
            new_dx = sin_theta * np.cos(phi)
            new_dy = sin_theta * np.sin(phi)
            new_dz = cos_theta * np.sign(dz)
        else:
            temp = np.sqrt(max(0, 1 - dz**2))
            new_dx = sin_theta * (dx * dz * np.cos(phi) - dy * np.sin(phi)) / temp + dx * cos_theta
            new_dy = sin_theta * (dy * dz * np.cos(phi) + dx * np.sin(phi)) / temp + dy * cos_theta
            new_dz = -sin_theta * np.cos(phi) * temp + dz * cos_theta

        norm = np.sqrt(new_dx**2 + new_dy**2 + new_dz**2)
        if norm > 0:
            return np.array([new_dx/norm, new_dy/norm, new_dz/norm])
        return direction

    def _fresnel_reflectance(self, n1: float, n2: float, cos_theta: float) -> float:
        """Calculate Fresnel reflectance at interface"""
        sin_theta = np.sqrt(max(0, 1 - cos_theta**2))
        sin_theta_t = n1 * sin_theta / n2

        if sin_theta_t >= 1.0:
            return 1.0  # Total internal reflection

        cos_theta_t = np.sqrt(max(0, 1 - sin_theta_t**2))

        rs = ((n1 * cos_theta - n2 * cos_theta_t) / (n1 * cos_theta + n2 * cos_theta_t + 1e-10))**2
        rp = ((n1 * cos_theta_t - n2 * cos_theta) / (n1 * cos_theta_t + n2 * cos_theta + 1e-10))**2

        return 0.5 * (rs + rp)

    def _get_layer_at_depth(self, z: float) -> Tuple[OpticalProperties, str]:
        """Get the tissue layer and properties at given depth"""

        if z < 0:
            return None, "escaped_top"

        if self.include_hair and z < self.z_epidermis_start:
            # In hair layer - probabilistic hit
            if np.random.random() < self.hair_coverage:
                return self.hair_props, "hair"
            else:
                return None, "air_gap"

        if z < self.z_dermis_start:
            return self.epidermis_props, "epidermis"

        if z < self.z_fat_start:
            return self.dermis_props, "dermis"

        return self.fat_props, "subcutaneous"

    def _record_energy(self, z: float, energy: float):
        """Record energy deposition at depth z"""
        idx = int(z / 0.5 * 100)
        if 0 <= idx < 100:
            self.energy_deposition[idx] += energy

    def _simulate_photon(self) -> Dict:
        """Simulate a single photon trajectory"""

        # Initial position at surface
        x, y, z = 0.0, 0.0, 0.0

        # Initial direction (perpendicular to surface)
        direction = np.array([0.0, 0.0, 1.0])

        # Initial weight
        weight = 1.0

        # Track maximum depth reached
        max_depth = 0.0

        # Track if passed through hair or gap
        went_through_hair = False
        went_through_gap = False

        max_steps = 10000

        for _ in range(max_steps):
            if weight < 1e-6:
                break

            props, layer = self._get_layer_at_depth(z)

            # Handle escaping
            if layer == "escaped_top":
                self.stats["reflected"] += 1
                break

            # Handle air gap (no interaction)
            if layer == "air_gap":
                went_through_gap = True
                step = 0.001  # Small step through air
                z += step * direction[2]
                continue

            if props is None or props.mu_t <= 0:
                z += 0.001 * direction[2]
                continue

            # Track hair passage
            if layer == "hair":
                went_through_hair = True

            # Sample step size
            step = self._sample_step_size(props.mu_t)

            # Calculate new position
            new_z = z + step * direction[2]
            new_x = x + step * direction[0]
            new_y = y + step * direction[1]

            # Check for boundary crossing
            if new_z < 0:
                self.stats["reflected"] += 1
                break

            # Update position
            x, y, z = new_x, new_y, new_z
            max_depth = max(max_depth, z)

            # Absorption
            absorbed_frac = props.mu_a / props.mu_t
            absorbed_weight = weight * absorbed_frac
            weight *= (1 - absorbed_frac)

            # Record energy deposition
            self._record_energy(z, absorbed_weight)

            # Track absorption by layer
            if layer == "hair":
                self.stats["absorbed_in_hair"] += absorbed_weight
            elif layer == "epidermis":
                self.stats["absorbed_in_epidermis"] += absorbed_weight
            elif layer == "dermis":
                self.stats["absorbed_in_dermis"] += absorbed_weight

            # Scattering
            direction = self._henyey_greenstein_scatter(direction, props.g)

            # Russian roulette
            if weight < 0.01:
                if np.random.random() < 0.1:
                    weight /= 0.1
                else:
                    break

        # Record statistics based on depth reached
        if max_depth >= self.z_dermis_start:
            self.stats["reached_dermis"] += 1
        if max_depth >= self.z_epidermis_start:
            self.stats["reached_epidermis"] += 1

        if went_through_hair:
            self.stats["transmitted_through_hair"] += 1
        elif went_through_gap:
            self.stats["passed_through_gap"] += 1

        return {
            "max_depth": max_depth,
            "final_weight": weight
        }

    def run(self) -> Dict:
        """Run the simulation"""

        print(f"\n{'='*60}")
        print(f"Monte Carlo Light Transmission Simulation (Improved Model)")
        print(f"{'='*60}")
        print(f"Wavelength: {self.wavelength} nm")
        print(f"Irradiance: {self.irradiance} mW/cm²")
        print(f"Photons: {self.num_photons:,}")
        print(f"Hair type: {self.hair_type}")
        print(f"Include hair: {self.include_hair}")
        print(f"Hair coverage: {self.hair_coverage*100:.1f}%")
        print(f"Hair μa: {self.hair_props.mu_a:.1f} cm⁻¹")
        print(f"{'='*60}\n")

        start_time = time.time()

        max_depths = []

        for i in range(self.num_photons):
            if (i + 1) % 20000 == 0:
                print(f"Progress: {i+1:,}/{self.num_photons:,} ({100*(i+1)/self.num_photons:.1f}%)")

            result = self._simulate_photon()
            max_depths.append(result["max_depth"])

        elapsed = time.time() - start_time

        # Calculate fluence at various depths
        fluence_data = self._calculate_fluence(max_depths)

        return self._compile_results(elapsed, fluence_data)

    def _calculate_fluence(self, depths: List[float]) -> Dict:
        """Calculate fluence at various depths"""
        depths = np.array(depths)

        fluence = {}
        depth_points = {
            "surface": 0.0,
            "epidermis_top": self.z_epidermis_start,
            "dermis_top": self.z_dermis_start,
            "dermis_100um": self.z_dermis_start + 0.01,
            "dermis_500um": self.z_dermis_start + 0.05,
            "dermis_1mm": self.z_dermis_start + 0.1,
            "dermis_2mm": self.z_dermis_start + 0.2,
        }

        for name, z in depth_points.items():
            fraction = np.mean(depths >= z)
            fluence[name] = {
                "depth_cm": z,
                "depth_um": z * 10000,
                "transmission_fraction": fraction,
                "fluence_mW_cm2": self.irradiance * fraction
            }

        return fluence

    def _compile_results(self, elapsed: float, fluence_data: Dict) -> Dict:
        """Compile simulation results"""

        n = self.num_photons

        results = {
            "parameters": {
                "wavelength_nm": self.wavelength,
                "irradiance_mW_cm2": self.irradiance,
                "num_photons": n,
                "hair_type": self.hair_type,
                "include_hair": self.include_hair,
                "hair_coverage_percent": self.hair_coverage * 100,
                "elapsed_seconds": elapsed
            },
            "optical_properties": {
                "hair": {
                    "mu_a_cm-1": self.hair_props.mu_a,
                    "mu_s_cm-1": self.hair_props.mu_s,
                    "g": self.hair_props.g
                },
                "epidermis": {
                    "mu_a_cm-1": self.epidermis_props.mu_a,
                    "mu_s_cm-1": self.epidermis_props.mu_s,
                    "g": self.epidermis_props.g
                },
                "dermis": {
                    "mu_a_cm-1": self.dermis_props.mu_a,
                    "mu_s_cm-1": self.dermis_props.mu_s,
                    "g": self.dermis_props.g
                }
            },
            "geometry": {
                "hair_diameter_um": self.hair.diameter * 10000,
                "hair_density_per_cm2": self.hair.density,
                "epidermis_thickness_um": self.epidermis_thickness * 10000,
                "dermis_thickness_um": self.dermis_thickness * 10000
            },
            "transmission_statistics": {
                "reached_epidermis_count": self.stats["reached_epidermis"],
                "reached_epidermis_percent": 100 * self.stats["reached_epidermis"] / n,
                "reached_dermis_count": self.stats["reached_dermis"],
                "reached_dermis_percent": 100 * self.stats["reached_dermis"] / n,
                "reflected_count": self.stats["reflected"],
                "reflected_percent": 100 * self.stats["reflected"] / n,
                "transmitted_through_hair_count": self.stats["transmitted_through_hair"],
                "transmitted_through_hair_percent": 100 * self.stats["transmitted_through_hair"] / n,
                "passed_through_gap_count": self.stats["passed_through_gap"],
                "passed_through_gap_percent": 100 * self.stats["passed_through_gap"] / n
            },
            "absorption_distribution": {
                "absorbed_in_hair": self.stats["absorbed_in_hair"],
                "absorbed_in_hair_percent": 100 * self.stats["absorbed_in_hair"] / n,
                "absorbed_in_epidermis": self.stats["absorbed_in_epidermis"],
                "absorbed_in_epidermis_percent": 100 * self.stats["absorbed_in_epidermis"] / n,
                "absorbed_in_dermis": self.stats["absorbed_in_dermis"],
                "absorbed_in_dermis_percent": 100 * self.stats["absorbed_in_dermis"] / n
            },
            "fluence_at_depths": fluence_data,
            "energy_deposition": {
                "depth_bins_cm": self.depth_bins.tolist(),
                "deposition": self.energy_deposition.tolist()
            }
        }

        return results


def run_comprehensive_study():
    """Run comprehensive comparative study"""

    print("\n" + "="*80)
    print("COMPREHENSIVE LIGHT TRANSMISSION STUDY")
    print("630nm LED (2 mW/cm²) Through Hair to Scalp")
    print("="*80)

    all_results = {}

    # Baseline: No hair
    print("\n>>> BASELINE: Direct skin contact (no hair)")
    sim = ImprovedScalpSimulation(num_photons=100000, include_hair=False)
    all_results["no_hair"] = sim.run()

    # Different hair types
    hair_types = ["black", "dark_brown", "light_brown", "blonde", "gray"]

    for hair_type in hair_types:
        print(f"\n>>> Testing: {hair_type.upper()} hair")
        sim = ImprovedScalpSimulation(
            num_photons=100000,
            hair_type=hair_type,
            include_hair=True
        )
        all_results[hair_type] = sim.run()

    return all_results


def print_results_summary(results: Dict):
    """Print formatted results summary"""

    print("\n\n")
    print("="*100)
    print("SIMULATION RESULTS SUMMARY")
    print("630nm Red LED | 2 mW/cm² Irradiance | Monte Carlo (100,000 photons)")
    print("="*100)

    # Table header
    print(f"\n{'Condition':<15} {'Hair Coverage':<14} {'Dermis %':<12} {'Fluence@Dermis':<18} {'Hair Abs %':<12}")
    print("-"*75)

    # No hair baseline
    no_hair = results["no_hair"]
    dermis_pct = no_hair["transmission_statistics"]["reached_dermis_percent"]
    fluence = no_hair["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"]
    print(f"{'No Hair':<15} {'N/A':<14} {dermis_pct:>8.1f}% {fluence:>14.4f} mW/cm² {'N/A':>10}")

    # Hair types
    for hair_type in ["black", "dark_brown", "light_brown", "blonde", "gray"]:
        r = results[hair_type]
        coverage = r["parameters"]["hair_coverage_percent"]
        dermis_pct = r["transmission_statistics"]["reached_dermis_percent"]
        fluence = r["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"]
        hair_abs = r["absorption_distribution"]["absorbed_in_hair_percent"]

        print(f"{hair_type.title():<15} {coverage:>10.1f}% {dermis_pct:>10.1f}% {fluence:>14.4f} mW/cm² {hair_abs:>9.1f}%")

    print("-"*75)

    # Detailed depth analysis for dark brown
    print("\n\nDETAILED DEPTH PROFILE (Dark Brown Hair):")
    print("-"*60)

    if "dark_brown" in results:
        r = results["dark_brown"]
        print(f"{'Depth Location':<20} {'Transmission %':<18} {'Fluence (mW/cm²)':<18}")
        print("-"*56)

        for name, data in r["fluence_at_depths"].items():
            depth_um = data["depth_um"]
            trans = data["transmission_fraction"] * 100
            fluence = data["fluence_mW_cm2"]
            print(f"{name:<20} {trans:>14.2f}% {fluence:>14.4f}")

    # Analysis and conclusions
    print("\n\n" + "="*100)
    print("KEY FINDINGS AND CLINICAL IMPLICATIONS")
    print("="*100)

    if "dark_brown" in results:
        dark = results["dark_brown"]
        no_hair_dermis = results["no_hair"]["transmission_statistics"]["reached_dermis_percent"]
        dark_dermis = dark["transmission_statistics"]["reached_dermis_percent"]
        reduction = (1 - dark_dermis / no_hair_dermis) * 100 if no_hair_dermis > 0 else 0

        fluence_at_dermis = dark["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"]
        hair_absorbed = dark["absorption_distribution"]["absorbed_in_hair_percent"]

        # Therapeutic calculations
        therapeutic_min = 1.0  # mW/cm²
        therapeutic_max = 5.0  # mW/cm²

        # Time to reach therapeutic dose (J/cm² = mW/cm² × seconds / 1000)
        target_dose = 3.0  # J/cm² typical therapeutic dose
        if fluence_at_dermis > 0:
            time_to_dose_seconds = (target_dose * 1000) / fluence_at_dermis
            time_to_dose_minutes = time_to_dose_seconds / 60
        else:
            time_to_dose_minutes = float('inf')

        print(f"""
1. LIGHT PENETRATION ANALYSIS:
   - Hair coverage: {dark["parameters"]["hair_coverage_percent"]:.1f}%
   - Light reduction due to hair: {reduction:.1f}%
   - Hair absorption: {hair_absorbed:.1f}% of incident light

2. FLUENCE AT TARGET DEPTH (Dermis 500μm):
   - With dark brown hair: {fluence_at_dermis:.4f} mW/cm²
   - Original surface irradiance: 2.0 mW/cm²
   - Transmission efficiency: {fluence_at_dermis/2.0*100:.2f}%

3. THERAPEUTIC ASSESSMENT:
   - Minimum therapeutic irradiance: ~{therapeutic_min}-{therapeutic_max} mW/cm²
   - Current fluence at dermis: {fluence_at_dermis:.4f} mW/cm²
   - Status: {'WITHIN' if therapeutic_min <= fluence_at_dermis <= therapeutic_max else 'BELOW' if fluence_at_dermis < therapeutic_min else 'ABOVE'} therapeutic range

4. TREATMENT TIME CALCULATION:
   - Target dose: {target_dose} J/cm² (typical photobiomodulation)
   - Estimated treatment time: {time_to_dose_minutes:.1f} minutes

5. HAIR COLOR IMPACT (at dermis 500μm):""")

        for ht in ["black", "dark_brown", "light_brown", "blonde", "gray"]:
            fl = results[ht]["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"]
            print(f"   - {ht.title():<12}: {fl:.4f} mW/cm²")

        print(f"""
6. CLINICAL RECOMMENDATIONS:
   - 630nm light DOES penetrate through hair to reach dermis
   - Dark hair absorbs more light but transmission still occurs
   - For optimal results with dark hair:
     * Increase treatment time (compensate for absorption)
     * Use higher irradiance LEDs (4-10 mW/cm²)
     * Part hair to maximize direct skin contact
   - Gray/blonde hair allows significantly more transmission
""")

    return results


def main():
    """Main entry point"""

    # Run comprehensive study
    results = run_comprehensive_study()

    # Print summary
    print_results_summary(results)

    # Save to JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    output_file = "simulation_results_v2.json"
    with open(output_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = main()
