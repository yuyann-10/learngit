import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import diffrax 
from diffrax import diffeqsolve, PIDController, ODETerm, Dopri5, SaveAt
import matplotlib
import os
import gc 
import equinox as eqx

# --- 全局设置 ---
matplotlib.use('TkAgg') 
jax.config.update("jax_enable_x64", True)

# --- 常量定义 ---
mu = 0.01215058 
L = 389703e3         # m
vv = 1017.551785     # m/s
tt = 382981          # s
tt_days = tt / (24 * 3600) 
M = 5.9722e24 + 7.346e22 
mu3 = 1.989e30 / M  
R3 = 149.6e9 / L    
n_s = 0.9252         

# -------------------------------------------------------------
#                    BCR4BP 动力学
# -------------------------------------------------------------
@jax.jit
def bcr4bp_rhs(t, state, args):
    x, y, z, vx, vy, vz = state
    mu, mu3, R3, n_s = args["mu"], args["mu3"], args["R3"], args["n_s"]
    x_moon = 1.0 - mu
    r_moon_norm = 1738.0 / 389703.0 
    
    dist_to_moon_sq = (x - x_moon)**2 + y**2 + z**2
    
    def derivatives(_):
        mu1 = 1.0 - mu
        r1_sq = (x + mu)**2 + y**2 + z**2 + 1e-18
        r2_sq = (x - mu1)**2 + y**2 + z**2 + 1e-18
        inv_r1 = jax.lax.rsqrt(r1_sq)
        inv_r1_3 = inv_r1**3
        inv_r2 = jax.lax.rsqrt(r2_sq)
        inv_r2_3 = inv_r2**3
        
        psi = n_s * t 
        R_sun_x, R_sun_y = R3 * jnp.cos(psi), R3 * jnp.sin(psi)
        r3_sq = (x - R_sun_x)**2 + (y - R_sun_y)**2 + z**2 + 1e-18
        inv_r3 = jax.lax.rsqrt(r3_sq)
        inv_r3_3 = inv_r3**3
        
        ax_cr3bp = x + 2.0 * vy - mu1 * (x + mu) * inv_r1_3 - mu * (x - mu1) * inv_r2_3 
        ay_cr3bp = y - 2.0 * vx - mu1 * y * inv_r1_3 - mu * y * inv_r2_3 
        az_cr3bp = -mu1 * z * inv_r1_3 - mu * z * inv_r2_3 
        
        ax_sun = -mu3 * (x - R_sun_x) * inv_r3_3 - mu3 * jnp.cos(psi) / R3**2
        ay_sun = -mu3 * (y - R_sun_y) * inv_r3_3 - mu3 * jnp.sin(psi) / R3**2
        az_sun = -mu3 * z * inv_r3_3
        
        return jnp.array([vx, vy, vz, ax_cr3bp + ax_sun, ay_cr3bp + ay_sun, az_cr3bp + az_sun])

    def frozen(_):
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return jax.lax.cond(dist_to_moon_sq < r_moon_norm**2, frozen, derivatives, None)

@eqx.filter_jit
def propagate_segments(X_init, args, duration_days, num_steps):
    save_ts = jnp.linspace(0, duration_days / tt_days, num_steps)
    base_term = ODETerm(bcr4bp_rhs)
    solver = Dopri5()
    stepsize_controller = PIDController(rtol=1e-7, atol=1e-9)
    sol = diffeqsolve(
        base_term, solver, t0=0.0, t1=duration_days / tt_days, 
        y0=X_init, dt0=1e-3, saveat=SaveAt(ts=save_ts), 
        args=args, stepsize_controller=stepsize_controller,
        max_steps=20000, throw=False
    )
    return sol.ys 

@jax.jit
def extract_batch_components(trajs_batch, sc_pos_batch, sc_vel_batch, 
                               area_sphere_km2, R_DZ_lu, vv_kms, V_DZ_km3):
    rel_pos = trajs_batch[:, :, 0:3] - sc_pos_batch[None, :, :]
    in_dz_mask = jnp.sum(rel_pos**2, axis=-1) < R_DZ_lu**2
    N_in_dz_t = jnp.sum(in_dz_mask, axis=0)

    v_rel_vu = trajs_batch[:, :, 3:6] - sc_vel_batch[None, :, :]
    v_total_kms = jnp.linalg.norm(v_rel_vu, axis=-1) * vv_kms
    
    # 通量计算: (相对速度 * 截面积) / 危险区体积
    e_rate_per_fragment = v_total_kms * area_sphere_km2 / V_DZ_km3 
    Sum_E_Rate_Flux_t = jnp.sum(e_rate_per_fragment * in_dz_mask, axis=0)  
    
    return N_in_dz_t, Sum_E_Rate_Flux_t

# -------------------------------------------------------------
#                           Main
# -------------------------------------------------------------
def main():
    bcr4bp_args = {"mu": mu, "mu3": mu3, "R3": R3, "n_s": n_s}
    
    # 1. 设定解体点位置 (LU) - 直接硬编码
    pos_breakup_nd = jnp.array([1.06205353630756,	0.00119514047891347,	0.00152897141216298])

    # 2. 加载碎片速度属性 
    file_name = 'fragmentr_properties.txt'
    df_frags = pd.read_csv(file_name, sep=',', comment='#', header=None,
                           names=['ID', 'VX_abs', 'VY_abs', 'VZ_abs', 'Vmag', 'Mass'])
    v_data = df_frags[['VX_abs', 'VY_abs', 'VZ_abs']].values
    N_fragments = len(v_data)

    # 3. 加载 Artemis 轨道
    df_artemis = pd.read_csv("artemis_bcr4bp_unified.csv")
    artemis_times = df_artemis['Time_Days'].values
    artemis_states = df_artemis[['x','y','z','vx','vy','vz']].values
    sc_interp = interp1d(artemis_times, artemis_states, axis=0, bounds_error=False, fill_value="extrapolate")
    TOTAL_END = np.max(artemis_times)

    # 设定爆炸时刻  
    all_t_exps = np.linspace(0.0, TOTAL_END - 2.0, 8) 
    t_exp = all_t_exps[4] 

    # 风险评估参数
    R_DZ_km = 100.0
    R_DZ_lu = R_DZ_km / (L / 1000.0)
    V_DZ_km3 = (4/3) * np.pi * (R_DZ_km**3)
    AREA_SC_KM2 = jnp.pi * (0.01**2) # 10m 半径航天器
    vv_kms = vv / 1000.0
    
    # 准备碎片初值 (N, 6)
    frags_init = jnp.stack([
        jnp.full(N_fragments, pos_breakup_nd[0]), 
        jnp.full(N_fragments, pos_breakup_nd[1]), 
        jnp.full(N_fragments, pos_breakup_nd[2]),
        v_data[:,0]/vv, v_data[:,1]/vv, v_data[:,2]/vv
    ], axis=1)

    # 评估时间跨度 (从爆炸那一刻起)
    eval_duration = TOTAL_END - t_exp
    NUM_STEPS_EVAL = 20000
    T_EVAL_REL = np.linspace(0, eval_duration, NUM_STEPS_EVAL)
    dt_sec = (T_EVAL_REL[1] - T_EVAL_REL[0]) * 24 * 3600
    
    # 同步获取航天器轨迹
    SC_DATA = sc_interp(t_exp + T_EVAL_REL)
    SC_POS_JAX = jnp.array(SC_DATA[:, 0:3])
    SC_VEL_JAX = jnp.array(SC_DATA[:, 3:6])

    # 批处理计算
    global_Sum_Flux = np.zeros(NUM_STEPS_EVAL)
    global_N_in_dz = np.zeros(NUM_STEPS_EVAL)
    BATCH_SIZE = 2000
    num_batches = N_fragments // BATCH_SIZE
    
    vmap_prop = eqx.filter_jit(jax.vmap(propagate_segments, in_axes=(0, None, None, None)))

    print(f"\n评估开始 | 爆炸时刻: {t_exp:.2f} d | 持续时长: {eval_duration:.2f} d")
    
    for b_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start, end = b_idx * BATCH_SIZE, (b_idx + 1) * BATCH_SIZE
        fb = frags_init[start:end]
        
        # 直接传播评估全过程
        eval_trajs = vmap_prop(fb, bcr4bp_args, eval_duration, NUM_STEPS_EVAL)
        eval_trajs = jnp.where(jnp.isnan(eval_trajs), 1e6, eval_trajs) # 异常处理
        
        # 计算风险组件
        n_t, flux_t = extract_batch_components(
            eval_trajs, SC_POS_JAX, SC_VEL_JAX, 
            AREA_SC_KM2, R_DZ_lu, vv_kms, V_DZ_km3
        )
        global_N_in_dz += np.array(n_t)
        global_Sum_Flux += np.array(flux_t)
        current_batch_n = np.array(n_t)
        hit_indices = np.where(current_batch_n > 0)[0]
 
        if len(hit_indices) > 0:
            for idx in hit_indices[::50]:
                # 打印当前时刻 (t_exp 是爆炸时刻, T_EVAL_REL[idx] 是相对于爆炸的偏移)
                current_time_days = t_exp + T_EVAL_REL[idx]
                num_frags = current_batch_n[idx]
                print(f"检测到碎片进入! 时刻: {current_time_days:.4f} days | 进入数量: {int(num_frags)}")
    # 最终结果
    E_total = np.sum(global_Sum_Flux * dt_sec)
    P_hazard = 1.0 - np.exp(-E_total)
    
    print(f"\n结果汇总:")
    print(f"解体位置: {pos_breakup_nd}")
    print(f"危险区内碎片峰值数: {np.max(global_N_in_dz)}")
    print(f"总期望碰撞数 (E): {E_total:.4e}")
    print(f"总失效概率 (P): {P_hazard:.4e}")

    # 保存
    np.savez_compressed("artemis_hazard_direct_p5.npz", 
                        time=T_EVAL_REL, 
                        flux=global_Sum_Flux, 
                        n_dz=global_N_in_dz)

if __name__ == "__main__":
    main()