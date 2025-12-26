#!/usr/bin/env python3
"""
Visualization for Light Transmission Simulation Results
========================================================
Generates plots and summary for the Monte Carlo simulation results.
"""

import json
import numpy as np

def load_results(filename: str = "simulation_results_v2.json"):
    """Load simulation results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def print_detailed_report(results: dict):
    """Print a detailed report of simulation results"""

    print("\n" + "="*80)
    print("   630nm LED 광전달 시뮬레이션 결과 보고서")
    print("   Light Transmission Through Hair to Scalp - Simulation Report")
    print("="*80)

    # Simulation parameters
    print("\n[1] 시뮬레이션 파라미터 (Simulation Parameters)")
    print("-"*60)
    params = results["no_hair"]["parameters"]
    print(f"   파장 (Wavelength):        {params['wavelength_nm']} nm")
    print(f"   조사량 (Irradiance):      {params['irradiance_mW_cm2']} mW/cm²")
    print(f"   광자 수 (Photon count):   {params['num_photons']:,}")

    # Geometry
    print("\n[2] 기하학적 파라미터 (Geometric Parameters)")
    print("-"*60)
    geom = results["dark_brown"]["geometry"]
    print(f"   모발 직경 (Hair diameter):     {geom['hair_diameter_um']:.1f} μm")
    print(f"   모발 밀도 (Hair density):      {geom['hair_density_per_cm2']:.0f} follicles/cm²")
    print(f"   표피 두께 (Epidermis):         {geom['epidermis_thickness_um']:.0f} μm")
    print(f"   진피 두께 (Dermis):            {geom['dermis_thickness_um']:.0f} μm")

    # Optical properties
    print("\n[3] 광학적 특성 at 630nm (Optical Properties)")
    print("-"*60)

    opt = results["dark_brown"]["optical_properties"]
    print(f"   {'조직':<15} {'흡수계수 μa':<18} {'산란계수 μs':<18} {'비등방성 g':<12}")
    print(f"   {'Tissue':<15} {'(cm⁻¹)':<18} {'(cm⁻¹)':<18}")
    print("   " + "-"*55)

    for tissue, name_ko in [("hair", "모발"), ("epidermis", "표피"), ("dermis", "진피")]:
        props = opt[tissue]
        print(f"   {name_ko:<15} {props['mu_a_cm-1']:<18.1f} {props['mu_s_cm-1']:<18.1f} {props['g']:<12.2f}")

    # Transmission results
    print("\n[4] 광투과 결과 (Light Transmission Results)")
    print("-"*60)
    print(f"\n   {'모발 색상':<12} {'진피 도달률':<15} {'500μm 플루언스':<18} {'모발 흡수':<12}")
    print(f"   {'Hair Type':<12} {'Dermis Reach':<15} {'Fluence@500μm':<18} {'Hair Abs.':<12}")
    print("   " + "-"*55)

    # No hair baseline
    no_hair = results["no_hair"]
    dermis_pct = no_hair["transmission_statistics"]["reached_dermis_percent"]
    fluence = no_hair["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"]
    print(f"   {'없음 (None)':<12} {dermis_pct:>10.1f}% {fluence:>14.4f} mW/cm² {'N/A':>10}")

    for hair_type, name_ko in [("black", "검정"), ("dark_brown", "짙은갈색"),
                                ("light_brown", "연한갈색"), ("blonde", "금발"),
                                ("gray", "회색")]:
        r = results[hair_type]
        dermis_pct = r["transmission_statistics"]["reached_dermis_percent"]
        fluence = r["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"]
        hair_abs = r["absorption_distribution"]["absorbed_in_hair_percent"]
        print(f"   {name_ko:<12} {dermis_pct:>10.1f}% {fluence:>14.4f} mW/cm² {hair_abs:>9.1f}%")

    # Depth profile for dark brown
    print("\n[5] 깊이별 투과 프로파일 - 짙은갈색 모발 (Depth Profile - Dark Brown)")
    print("-"*60)

    fluence_data = results["dark_brown"]["fluence_at_depths"]
    print(f"\n   {'깊이 위치':<20} {'깊이 (μm)':<12} {'투과율':<12} {'플루언스':<15}")
    print("   " + "-"*55)

    depth_names = {
        "surface": "표면",
        "epidermis_top": "표피 상단",
        "dermis_top": "진피 상단",
        "dermis_100um": "진피 100μm",
        "dermis_500um": "진피 500μm",
        "dermis_1mm": "진피 1mm",
        "dermis_2mm": "진피 2mm"
    }

    for key, data in fluence_data.items():
        name = depth_names.get(key, key)
        depth = data["depth_um"]
        trans = data["transmission_fraction"] * 100
        fluence = data["fluence_mW_cm2"]
        print(f"   {name:<20} {depth:>8.0f} {trans:>10.1f}% {fluence:>12.4f} mW/cm²")

    # Therapeutic analysis
    print("\n[6] 치료적 분석 (Therapeutic Analysis)")
    print("-"*60)

    dark_brown = results["dark_brown"]
    fluence_dermis = dark_brown["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"]

    print(f"""
   목표 조직: 진피 (Target tissue: Dermis)
   목표 깊이: 500 μm

   표면 조사량: 2.0 mW/cm²
   진피 도달 플루언스: {fluence_dermis:.4f} mW/cm²
   투과 효율: {fluence_dermis/2.0*100:.1f}%

   광생체조절 권장 조사량: 1-5 mW/cm²
   현재 진피 플루언스 상태: {'권장 범위 내' if 1.0 <= fluence_dermis <= 5.0 else '권장 범위 미만' if fluence_dermis < 1.0 else '권장 범위 초과'}

   목표 치료 용량: 3 J/cm² (일반적인 광생체조절 용량)
   필요 치료 시간: {3000/fluence_dermis/60:.1f} 분

   ※ 참고: 1 J/cm² = 1000 mW·s/cm²
""")

    # Key conclusions
    print("[7] 주요 결론 (Key Conclusions)")
    print("-"*60)
    print("""
   1. 630nm LED 빛은 모발을 통과하여 두피 진피층에 도달합니다.
      (630nm LED light DOES penetrate through hair to reach the dermis)

   2. 모발 색상에 따른 흡수 차이:
      - 검정 모발: 가장 높은 흡수 (~6%)
      - 회색/금발 모발: 낮은 흡수 (~0.5-1%)

   3. 표피층(150μm)에서 약 35-40%의 빛이 흡수/산란됩니다.

   4. 진피 500μm 깊이에서 약 39%의 빛이 투과합니다.

   5. 2 mW/cm² 조사량으로 치료 용량(3 J/cm²) 달성까지 약 64분이 소요됩니다.

   6. 권장사항:
      - 더 높은 조사량(4-10 mW/cm²) LED 사용 고려
      - 모발을 가르거나 접촉 면적 최대화
      - 짙은 모발의 경우 치료 시간 연장
""")

    return results


def create_ascii_depth_chart(results: dict):
    """Create ASCII art depth profile chart"""

    print("\n[8] 깊이별 투과율 그래프 (Depth Transmission Chart)")
    print("-"*60)

    dark_brown = results["dark_brown"]
    fluence_data = dark_brown["fluence_at_depths"]

    # Sort by depth
    sorted_depths = sorted(fluence_data.items(), key=lambda x: x[1]["depth_cm"])

    print("\n   투과율 (Transmission %)")
    print("   0%        25%        50%        75%       100%")
    print("   |----------|----------|----------|----------|")

    for name, data in sorted_depths:
        trans_pct = data["transmission_fraction"] * 100
        bar_length = int(trans_pct / 2)  # Scale to fit 50 chars
        bar = "█" * bar_length + "░" * (50 - bar_length)
        depth_um = data["depth_um"]

        # Shorten names for display
        display_names = {
            "surface": "표면     ",
            "epidermis_top": "표피상단 ",
            "dermis_top": "진피상단 ",
            "dermis_100um": "진피100μm",
            "dermis_500um": "진피500μm",
            "dermis_1mm": "진피1mm  ",
            "dermis_2mm": "진피2mm  "
        }

        name_display = display_names.get(name, name[:9])
        print(f"   {name_display} |{bar}| {trans_pct:.1f}%")

    print()


def create_hair_comparison_chart(results: dict):
    """Create ASCII chart comparing hair types"""

    print("\n[9] 모발 색상별 진피 도달 플루언스 비교")
    print("-"*60)

    print("\n   플루언스 at 진피 500μm (mW/cm²)")
    print("   0.0       0.2       0.4       0.6       0.8       1.0")
    print("   |---------|---------|---------|---------|---------|")

    hair_types = [
        ("no_hair", "모발없음"),
        ("black", "검정    "),
        ("dark_brown", "짙은갈색"),
        ("light_brown", "연한갈색"),
        ("blonde", "금발    "),
        ("gray", "회색    ")
    ]

    for hair_key, name in hair_types:
        fluence = results[hair_key]["fluence_at_depths"]["dermis_500um"]["fluence_mW_cm2"]
        bar_length = int(fluence * 50)  # Scale: 0-1 mW/cm² -> 0-50 chars
        bar = "▓" * bar_length + "░" * (50 - bar_length)
        print(f"   {name}   |{bar}| {fluence:.4f}")

    print()


def main():
    """Main function"""

    # Load results
    results = load_results()

    # Print detailed report
    print_detailed_report(results)

    # Create ASCII charts
    create_ascii_depth_chart(results)
    create_hair_comparison_chart(results)

    print("\n" + "="*80)
    print("   시뮬레이션 완료 - Simulation Complete")
    print("   결과 파일: simulation_results_v2.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
