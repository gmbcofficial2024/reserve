#!/usr/bin/env python3
"""
Monte Carlo Simulation for Light Transmission Through Hair and Scalp
======================================================================
Simulates 630nm LED light (2mW/cm²) penetration through hair follicles to scalp tissue.

Author: Light Transmission Simulation
Date: 2024
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict
import time

# Set random seed for reproducibility
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
        """Total interaction coefficient"""
        return self.mu_a + self.mu_s

    @property
    def mu_s_prime(self) -> float:
        """Reduced scattering coefficient"""
        return self.mu_s * (1 - self.g)

    @property
    def albedo(self) -> float:
        """Single scattering albedo"""
        return self.mu_s / self.mu_t


@dataclass
class GeometryParams:
    """Geometric parameters for hair and scalp"""
    hair_diameter: float  # Hair diameter (cm)
    hair_length: float    # Hair length within simulation volume (cm)
    hair_density: float   # Hair follicles per cm²
    epidermis_thickness: float  # Epidermis thickness (cm)
    dermis_thickness: float     # Dermis thickness (cm)


class ScalpLightSimulation:
    """
    Monte Carlo simulation for light transmission through hair to scalp.

    Tissue layers (from top to bottom):
    1. Air (source)
    2. Hair shaft (cylindrical geometry)
    3. Epidermis
    4. Dermis (target for light therapy)
    """

    def __init__(self, num_photons: int = 100000,
                 hair_type: str = "dark_brown",
                 include_hair: bool = True):
        """
        Initialize simulation parameters.

        Args:
            num_photons: Number of photons to simulate
            hair_type: Type of hair ("black", "dark_brown", "light_brown", "blonde", "gray")
            include_hair: Whether to include hair in simulation (for comparison)
        """
        self.num_photons = num_photons
        self.hair_type = hair_type
        self.include_hair = include_hair

        # LED source parameters
        self.wavelength = 630  # nm
        self.irradiance = 2.0  # mW/cm²

        # Initialize optical properties based on hair type
        self._init_optical_properties()
        self._init_geometry()

        # Statistics tracking
        self.stats = {
            "transmitted_to_epidermis": 0,
            "transmitted_to_dermis": 0,
            "absorbed_in_hair": 0,
            "absorbed_in_epidermis": 0,
            "absorbed_in_dermis": 0,
            "reflected": 0,
            "total_photons": num_photons
        }

        # Energy deposition tracking (depth profile)
        self.depth_bins = np.linspace(0, 0.5, 101)  # 0 to 5mm depth, 50μm resolution
        self.energy_deposition = np.zeros(100)
        self.fluence_at_depths = {}

    def _init_optical_properties(self):
        """Initialize optical properties for each tissue layer at 630nm"""

        # Hair melanin absorption varies by hair color
        # Reference: Jacques SL, "Optical properties of biological tissues" (2013)
        hair_absorption = {
            "black": 8.0,      # High melanin
            "dark_brown": 5.0,
            "light_brown": 3.0,
            "blonde": 1.5,
            "gray": 0.8,       # Low melanin
            "white": 0.3
        }

        mu_a_hair = hair_absorption.get(self.hair_type, 5.0)

        # Hair optical properties at 630nm
        self.hair_props = OpticalProperties(
            mu_a=mu_a_hair,     # cm^-1 (varies with melanin)
            mu_s=250.0,         # cm^-1 (keratin scattering)
            g=0.9,              # Forward scattering
            n=1.55              # Refractive index of keratin
        )

        # Epidermis optical properties at 630nm
        # Reference: Bashkatov et al., "Optical properties of skin" (2011)
        self.epidermis_props = OpticalProperties(
            mu_a=2.5,           # cm^-1 (melanin + blood)
            mu_s=400.0,         # cm^-1
            g=0.8,              #
            n=1.4               #
        )

        # Dermis optical properties at 630nm
        self.dermis_props = OpticalProperties(
            mu_a=0.5,           # cm^-1 (mostly blood absorption)
            mu_s=250.0,         # cm^-1 (collagen scattering)
            g=0.9,              # Forward scattering
            n=1.4               #
        )

        # Air properties
        self.air_props = OpticalProperties(
            mu_a=0.0,
            mu_s=0.0,
            g=0.0,
            n=1.0
        )

    def _init_geometry(self):
        """Initialize geometric parameters"""
        self.geometry = GeometryParams(
            hair_diameter=0.008,        # 80 μm = 0.008 cm
            hair_length=0.1,            # 1 mm hair shaft in contact zone
            hair_density=150,           # 150 follicles/cm² (typical scalp)
            epidermis_thickness=0.015,  # 150 μm
            dermis_thickness=0.2        # 2 mm
        )

        # Calculate hair coverage fraction
        hair_cross_section = np.pi * (self.geometry.hair_diameter / 2) ** 2
        self.hair_coverage = min(0.95, self.geometry.hair_density * hair_cross_section)

        # Layer boundaries (z-coordinates, surface = 0)
        if self.include_hair:
            self.z_hair_top = 0.0
            self.z_hair_bottom = self.geometry.hair_length
        else:
            self.z_hair_top = None
            self.z_hair_bottom = 0.0

        self.z_epidermis_top = self.z_hair_bottom if self.include_hair else 0.0
        self.z_epidermis_bottom = self.z_epidermis_top + self.geometry.epidermis_thickness
        self.z_dermis_top = self.z_epidermis_bottom
        self.z_dermis_bottom = self.z_dermis_top + self.geometry.dermis_thickness

    def _get_step_size(self, props: OpticalProperties) -> float:
        """Sample step size from exponential distribution"""
        if props.mu_t == 0:
            return float('inf')
        return -np.log(np.random.random()) / props.mu_t

    def _scatter(self, direction: np.ndarray, g: float) -> np.ndarray:
        """
        Calculate new direction after scattering using Henyey-Greenstein phase function.

        Args:
            direction: Current direction vector [dx, dy, dz]
            g: Anisotropy factor

        Returns:
            New direction vector
        """
        # Sample scattering angle using Henyey-Greenstein
        if abs(g) < 1e-6:
            cos_theta = 2 * np.random.random() - 1
        else:
            temp = (1 - g**2) / (1 - g + 2*g*np.random.random())
            cos_theta = (1 + g**2 - temp**2) / (2 * g)
            cos_theta = np.clip(cos_theta, -1, 1)

        sin_theta = np.sqrt(1 - cos_theta**2)

        # Sample azimuthal angle
        phi = 2 * np.pi * np.random.random()

        # Rotate direction
        dx, dy, dz = direction

        if abs(dz) > 0.99999:
            # Special case for near-vertical direction
            new_dx = sin_theta * np.cos(phi)
            new_dy = sin_theta * np.sin(phi)
            new_dz = cos_theta * np.sign(dz)
        else:
            temp = np.sqrt(1 - dz**2)
            new_dx = (sin_theta * (dx * dz * np.cos(phi) - dy * np.sin(phi)) / temp
                      + dx * cos_theta)
            new_dy = (sin_theta * (dy * dz * np.cos(phi) + dx * np.sin(phi)) / temp
                      + dy * cos_theta)
            new_dz = -sin_theta * np.cos(phi) * temp + dz * cos_theta

        # Normalize
        norm = np.sqrt(new_dx**2 + new_dy**2 + new_dz**2)
        return np.array([new_dx/norm, new_dy/norm, new_dz/norm])

    def _fresnel_reflectance(self, n1: float, n2: float, cos_i: float) -> float:
        """Calculate Fresnel reflectance for unpolarized light"""
        sin_i = np.sqrt(1 - cos_i**2)
        sin_t = n1 * sin_i / n2

        if sin_t >= 1.0:  # Total internal reflection
            return 1.0

        cos_t = np.sqrt(1 - sin_t**2)

        rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))**2
        rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i))**2

        return 0.5 * (rs + rp)

    def _get_layer_props(self, z: float) -> Tuple[OpticalProperties, str]:
        """Get optical properties for given depth z"""
        if self.include_hair and z < self.z_hair_bottom:
            # Check if photon is in hair or gap between hairs
            if np.random.random() < self.hair_coverage:
                return self.hair_props, "hair"
            else:
                return self.air_props, "air_gap"
        elif z < self.z_epidermis_bottom:
            return self.epidermis_props, "epidermis"
        elif z < self.z_dermis_bottom:
            return self.dermis_props, "dermis"
        else:
            return None, "escaped"

    def _simulate_photon(self) -> Dict:
        """Simulate a single photon trajectory"""
        # Initial position (at surface)
        x, y, z = 0.0, 0.0, 0.0

        # Initial direction (perpendicular to surface, into tissue)
        direction = np.array([0.0, 0.0, 1.0])

        # Initial weight
        weight = 1.0

        # Track photon fate
        fate = "unknown"
        max_depth = 0.0

        # Maximum number of steps to prevent infinite loops
        max_steps = 10000
        step_count = 0

        while weight > 1e-4 and step_count < max_steps:
            step_count += 1

            # Get current layer properties
            props, layer = self._get_layer_props(z)

            if layer == "escaped":
                if direction[2] > 0:  # Moving deeper
                    fate = "transmitted_beyond"
                else:
                    fate = "reflected"
                break

            if layer == "air_gap":
                # Move through air gap
                z += 0.01 * direction[2]  # Small step
                continue

            if props.mu_t == 0:
                z += 0.01 * direction[2]
                continue

            # Sample step size
            step = self._get_step_size(props)

            # Move photon
            new_z = z + step * direction[2]
            new_x = x + step * direction[0]
            new_y = y + step * direction[1]

            # Check for layer boundary crossing
            crossed_boundary = False
            boundary_z = None

            if self.include_hair and z < self.z_hair_bottom <= new_z:
                boundary_z = self.z_hair_bottom
                crossed_boundary = True
            elif z < self.z_epidermis_bottom <= new_z:
                boundary_z = self.z_epidermis_bottom
                crossed_boundary = True
            elif new_z < 0:  # Reflected out of tissue
                fate = "reflected"
                self.stats["reflected"] += 1
                break
            elif new_z > self.z_dermis_bottom:
                fate = "transmitted_beyond"
                break

            if crossed_boundary:
                # Handle boundary reflection/transmission
                t_boundary = (boundary_z - z) / (direction[2] + 1e-10)
                z = boundary_z
                x = x + t_boundary * direction[0]
                y = y + t_boundary * direction[1]
            else:
                x, y, z = new_x, new_y, new_z

            max_depth = max(max_depth, z)

            # Absorption (weight reduction)
            absorbed_fraction = props.mu_a / props.mu_t
            absorbed_weight = weight * absorbed_fraction
            weight *= (1 - absorbed_fraction)

            # Record energy deposition
            depth_idx = int(z / 0.5 * 100)
            if 0 <= depth_idx < 100:
                self.energy_deposition[depth_idx] += absorbed_weight

            # Track absorption by layer
            if layer == "hair":
                self.stats["absorbed_in_hair"] += absorbed_weight
            elif layer == "epidermis":
                self.stats["absorbed_in_epidermis"] += absorbed_weight
            elif layer == "dermis":
                self.stats["absorbed_in_dermis"] += absorbed_weight

            # Scattering (change direction)
            direction = self._scatter(direction, props.g)

            # Russian roulette for low weight photons
            if weight < 0.01:
                if np.random.random() < 0.1:
                    weight /= 0.1
                else:
                    break

        # Determine final fate based on max depth reached
        if max_depth >= self.z_dermis_top:
            self.stats["transmitted_to_dermis"] += 1
            fate = "reached_dermis"
        elif max_depth >= self.z_epidermis_top:
            self.stats["transmitted_to_epidermis"] += 1
            fate = "reached_epidermis"

        return {
            "fate": fate,
            "max_depth": max_depth,
            "final_weight": weight
        }

    def run_simulation(self) -> Dict:
        """Run the full Monte Carlo simulation"""
        print(f"\n{'='*60}")
        print(f"Monte Carlo Light Transmission Simulation")
        print(f"{'='*60}")
        print(f"Wavelength: {self.wavelength} nm")
        print(f"Irradiance: {self.irradiance} mW/cm²")
        print(f"Number of photons: {self.num_photons:,}")
        print(f"Hair type: {self.hair_type}")
        print(f"Include hair: {self.include_hair}")
        print(f"Hair coverage: {self.hair_coverage*100:.1f}%")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Reset statistics
        self.stats = {
            "transmitted_to_epidermis": 0,
            "transmitted_to_dermis": 0,
            "absorbed_in_hair": 0,
            "absorbed_in_epidermis": 0,
            "absorbed_in_dermis": 0,
            "reflected": 0,
            "total_photons": self.num_photons
        }
        self.energy_deposition = np.zeros(100)

        # Run simulation
        depths_reached = []
        for i in range(self.num_photons):
            if (i + 1) % 10000 == 0:
                print(f"Progress: {i+1:,}/{self.num_photons:,} photons ({100*(i+1)/self.num_photons:.1f}%)")
            result = self._simulate_photon()
            depths_reached.append(result["max_depth"])

        elapsed_time = time.time() - start_time

        # Calculate fluence at different depths
        self._calculate_fluence(depths_reached)

        # Compile results
        results = self._compile_results(elapsed_time)

        return results

    def _calculate_fluence(self, depths: List[float]):
        """Calculate fluence at various tissue depths"""
        depths = np.array(depths)

        # Key depth points
        depth_points = {
            "surface (0 μm)": 0.0,
            "hair_bottom (100 μm)": 0.01,
            "epidermis_top (100 μm)": self.z_epidermis_top,
            "epidermis_bottom (250 μm)": self.z_epidermis_bottom,
            "dermis_100μm": self.z_dermis_top + 0.01,
            "dermis_500μm": self.z_dermis_top + 0.05,
            "dermis_1mm": self.z_dermis_top + 0.1,
            "dermis_2mm": self.z_dermis_top + 0.2,
        }

        for name, z in depth_points.items():
            # Fraction of photons that reached at least this depth
            fraction = np.mean(depths >= z)
            # Fluence = irradiance × fraction transmitted
            fluence = self.irradiance * fraction
            self.fluence_at_depths[name] = {
                "depth_cm": z,
                "depth_um": z * 10000,
                "transmission_fraction": fraction,
                "fluence_mW_cm2": fluence
            }

    def _compile_results(self, elapsed_time: float) -> Dict:
        """Compile simulation results"""
        results = {
            "simulation_parameters": {
                "wavelength_nm": self.wavelength,
                "irradiance_mW_cm2": self.irradiance,
                "num_photons": self.num_photons,
                "hair_type": self.hair_type,
                "include_hair": self.include_hair,
                "hair_coverage_fraction": self.hair_coverage,
                "elapsed_time_seconds": elapsed_time
            },
            "optical_properties": {
                "hair": {
                    "mu_a": self.hair_props.mu_a,
                    "mu_s": self.hair_props.mu_s,
                    "g": self.hair_props.g,
                    "mu_s_prime": self.hair_props.mu_s_prime,
                    "n": self.hair_props.n
                },
                "epidermis": {
                    "mu_a": self.epidermis_props.mu_a,
                    "mu_s": self.epidermis_props.mu_s,
                    "g": self.epidermis_props.g,
                    "mu_s_prime": self.epidermis_props.mu_s_prime,
                    "n": self.epidermis_props.n
                },
                "dermis": {
                    "mu_a": self.dermis_props.mu_a,
                    "mu_s": self.dermis_props.mu_s,
                    "g": self.dermis_props.g,
                    "mu_s_prime": self.dermis_props.mu_s_prime,
                    "n": self.dermis_props.n
                }
            },
            "geometry": {
                "hair_diameter_um": self.geometry.hair_diameter * 10000,
                "hair_density_per_cm2": self.geometry.hair_density,
                "epidermis_thickness_um": self.geometry.epidermis_thickness * 10000,
                "dermis_thickness_um": self.geometry.dermis_thickness * 10000
            },
            "photon_fate_statistics": {
                "reached_epidermis": self.stats["transmitted_to_epidermis"],
                "reached_epidermis_percent": 100 * self.stats["transmitted_to_epidermis"] / self.num_photons,
                "reached_dermis": self.stats["transmitted_to_dermis"],
                "reached_dermis_percent": 100 * self.stats["transmitted_to_dermis"] / self.num_photons,
                "reflected": self.stats["reflected"],
                "reflected_percent": 100 * self.stats["reflected"] / self.num_photons
            },
            "absorption_distribution": {
                "absorbed_in_hair": self.stats["absorbed_in_hair"],
                "absorbed_in_hair_percent": 100 * self.stats["absorbed_in_hair"] / self.num_photons,
                "absorbed_in_epidermis": self.stats["absorbed_in_epidermis"],
                "absorbed_in_epidermis_percent": 100 * self.stats["absorbed_in_epidermis"] / self.num_photons,
                "absorbed_in_dermis": self.stats["absorbed_in_dermis"],
                "absorbed_in_dermis_percent": 100 * self.stats["absorbed_in_dermis"] / self.num_photons
            },
            "fluence_at_depths": self.fluence_at_depths,
            "depth_profile": {
                "depth_bins_cm": self.depth_bins.tolist(),
                "energy_deposition": self.energy_deposition.tolist()
            }
        }

        return results


def run_comparative_study():
    """Run comparative study with and without hair for different hair types"""
    print("\n" + "="*80)
    print("COMPARATIVE STUDY: Light Transmission Through Hair and Scalp")
    print("="*80)
    print(f"Wavelength: 630 nm (Red LED)")
    print(f"Irradiance: 2 mW/cm²")
    print("="*80 + "\n")

    all_results = {}

    # Hair types to test
    hair_types = ["black", "dark_brown", "light_brown", "blonde", "gray"]

    # Run simulation without hair (baseline)
    print("\n" + "-"*60)
    print("BASELINE: No Hair (Direct skin contact)")
    print("-"*60)

    sim_no_hair = ScalpLightSimulation(
        num_photons=50000,
        include_hair=False
    )
    results_no_hair = sim_no_hair.run_simulation()
    all_results["no_hair"] = results_no_hair

    # Run simulations for each hair type
    for hair_type in hair_types:
        print("\n" + "-"*60)
        print(f"Testing with {hair_type.upper()} hair")
        print("-"*60)

        sim = ScalpLightSimulation(
            num_photons=50000,
            hair_type=hair_type,
            include_hair=True
        )
        results = sim.run_simulation()
        all_results[hair_type] = results

    return all_results


def print_summary_report(all_results: Dict):
    """Print a summary report of all simulation results"""
    print("\n\n")
    print("="*100)
    print("SUMMARY REPORT: Light Transmission to Scalp at 630nm with 2mW/cm² LED")
    print("="*100)

    # Header
    print(f"\n{'Condition':<20} {'Dermis Reach %':<18} {'Fluence @ Dermis':<20} {'Hair Absorption %':<18}")
    print("-"*80)

    # No hair baseline
    no_hair = all_results["no_hair"]
    dermis_pct = no_hair["photon_fate_statistics"]["reached_dermis_percent"]
    fluence_dermis = no_hair["fluence_at_depths"].get("dermis_500μm", {}).get("fluence_mW_cm2", 0)
    print(f"{'No Hair (baseline)':<20} {dermis_pct:>14.1f}% {fluence_dermis:>16.4f} mW/cm² {'N/A':>16}")

    # Hair types
    for hair_type in ["black", "dark_brown", "light_brown", "blonde", "gray"]:
        if hair_type in all_results:
            results = all_results[hair_type]
            dermis_pct = results["photon_fate_statistics"]["reached_dermis_percent"]
            fluence_dermis = results["fluence_at_depths"].get("dermis_500μm", {}).get("fluence_mW_cm2", 0)
            hair_abs = results["absorption_distribution"]["absorbed_in_hair_percent"]
            print(f"{hair_type.title():<20} {dermis_pct:>14.1f}% {fluence_dermis:>16.4f} mW/cm² {hair_abs:>14.1f}%")

    print("-"*80)

    # Detailed depth profile for dark brown hair
    print("\n\nDETAILED DEPTH PROFILE (Dark Brown Hair):")
    print("-"*60)

    if "dark_brown" in all_results:
        fluence_data = all_results["dark_brown"]["fluence_at_depths"]
        print(f"{'Depth':<25} {'Transmission %':<18} {'Fluence (mW/cm²)':<18}")
        print("-"*60)
        for name, data in fluence_data.items():
            trans_pct = data["transmission_fraction"] * 100
            fluence = data["fluence_mW_cm2"]
            print(f"{name:<25} {trans_pct:>14.2f}% {fluence:>14.4f}")

    # Conclusions
    print("\n\n" + "="*100)
    print("KEY FINDINGS:")
    print("="*100)

    if "dark_brown" in all_results:
        dark_brown = all_results["dark_brown"]
        no_hair_dermis = all_results["no_hair"]["photon_fate_statistics"]["reached_dermis_percent"]
        dark_dermis = dark_brown["photon_fate_statistics"]["reached_dermis_percent"]
        reduction = (1 - dark_dermis/no_hair_dermis) * 100 if no_hair_dermis > 0 else 0

        dermis_fluence = dark_brown["fluence_at_depths"].get("dermis_500μm", {}).get("fluence_mW_cm2", 0)

        print(f"""
1. LIGHT PENETRATION THROUGH HAIR:
   - Dark brown hair reduces light transmission to dermis by approximately {reduction:.1f}%
   - Hair acts as a significant optical barrier due to melanin absorption

2. FLUENCE AT TARGET DEPTH (Dermis):
   - With dark brown hair: {dermis_fluence:.4f} mW/cm² at 500μm dermis depth
   - Minimum therapeutic dose for photobiomodulation: ~1-5 mW/cm²

3. HAIR COLOR IMPACT:
   - Black hair: Highest absorption, lowest transmission
   - Blonde/Gray hair: Lower absorption, higher transmission

4. CLINICAL IMPLICATIONS:
   - For effective scalp phototherapy with 630nm LED at 2mW/cm²:
     * Light DOES reach the dermis, but at reduced intensity
     * Darker hair requires longer exposure times or higher irradiance
     * Parting hair or using higher power LEDs recommended for optimal results

5. THERAPEUTIC THRESHOLD ANALYSIS:
   - Typical photobiomodulation effective dose: 1-4 J/cm²
   - With current fluence reaching dermis, treatment time needed varies by hair type
""")

    return all_results


def main():
    """Main function to run all simulations"""

    # Run comparative study
    all_results = run_comparative_study()

    # Print summary report
    print_summary_report(all_results)

    # Save results to JSON
    output_file = "simulation_results.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = main()
