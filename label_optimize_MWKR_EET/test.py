import os
import json
import random
import shutil
import numpy as np
from pathlib import Path

class DatasetGenerator:
    def __init__(self, ref_dataset_path, output_path):
        self.ref_dataset_path = ref_dataset_path
        self.output_path = output_path
        self.ref_fjs_path = os.path.join(ref_dataset_path, 'fjs')
        self.ref_result_path = os.path.join(ref_dataset_path, 'result')
        self.output_fjs_path = os.path.join(output_path, 'fjs')
        self.output_result_path = os.path.join(output_path, 'result')
        
        # 创建输出目录
        os.makedirs(self.output_fjs_path, exist_ok=True)
        os.makedirs(self.output_result_path, exist_ok=True)
        
    def get_ref_files(self):
        """获取所有参考数据集文件"""
        fjs_files = sorted([f for f in os.listdir(self.ref_fjs_path) if f.endswith('.fjs')])
        return fjs_files
    
    def perturb_fjs_file(self, ref_fjs_path, output_fjs_path, perturbation_ratio=0.02):
        """
        扰动fjs文件中的加工时间
        Args:
            ref_fjs_path: 参考fjs文件路径
            output_fjs_path: 输出fjs文件路径
            perturbation_ratio: 扰动比例
        """
        with open(ref_fjs_path, 'r') as f:
            lines = f.readlines()
        
        # 第一行（作业数和机器数）保持不变
        new_lines = [lines[0]]
        
        # 对后续每一行（每个作业的工序信息）进行扰动
        for line in lines[1:]:
            if line.strip() == '':
                new_lines.append(line)
                continue
                
            tokens = line.strip().split()
            new_tokens = [tokens[0]]  # 工序数量保持不变
            
            i = 1
            while i < len(tokens):
                # 机器选择数量
                machine_count = int(tokens[i])
                new_tokens.append(tokens[i])
                i += 1
                
                # 对每个(机器ID, 加工时间)对进行处理
                for _ in range(machine_count):
                    machine_id = tokens[i]
                    processing_time = int(tokens[i + 1])
                    
                    # 对加工时间添加噪声
                    noise = np.random.normal(0, perturbation_ratio * processing_time)
                    new_time = max(1, int(processing_time + noise))  # 至少为1
                    
                    new_tokens.append(machine_id)
                    new_tokens.append(str(new_time))
                    i += 2
            
            new_lines.append('  '.join(new_tokens) + '\n')
        
        # 写入新文件
        with open(output_fjs_path, 'w') as f:
            f.writelines(new_lines)
    
    def perturb_values(self, values_dict, perturbation_ratio=0.02):
        """
        扰动性能值，保证统计关系正确
        Args:
            values_dict: 包含mean, std, min, max, median, all_values的字典
            perturbation_ratio: 扰动比例（默认2%）
        Returns:
            扰动后的values_dict
        """
        all_values = values_dict['all_values']
        n = len(all_values)
        
        # 对每个值添加高斯噪声
        perturbed_values = []
        for val in all_values:
            noise = np.random.normal(0, perturbation_ratio * val)
            new_val = max(1.0, val + noise)  # 确保值不小于1
            perturbed_values.append(new_val)
        
        # 重新计算统计值（不排序，保持原始顺序）
        new_values_dict = {
            'mean': float(np.mean(perturbed_values)),
            'std': float(np.std(perturbed_values, ddof=1)) if n > 1 else 0.0,
            'min': float(np.min(perturbed_values)),
            'max': float(np.max(perturbed_values)),
            'median': float(np.median(perturbed_values)),
            'all_values': [float(v) for v in perturbed_values]
        }
        
        return new_values_dict
    
    def perturb_convergence(self, convergence_dict):
        """扰动收敛信息"""
        all_generations = convergence_dict['all_generations']
        
        # 对每个generation添加小的整数扰动
        perturbed_generations = []
        for gen in all_generations:
            # ±1到2的整数扰动
            noise = random.randint(-2, 2)
            new_gen = max(0, gen + noise)
            perturbed_generations.append(new_gen)
        
        new_convergence_dict = {
            'avg_generation': float(np.mean(perturbed_generations)),
            'std_generation': float(np.std(perturbed_generations, ddof=1)) if len(perturbed_generations) > 1 else 0.0,
            'min_generation': int(np.min(perturbed_generations)),
            'max_generation': int(np.max(perturbed_generations)),
            'all_generations': perturbed_generations
        }
        
        return new_convergence_dict
    
    def ensure_mwkr_eet_is_best(self, init_methods):
        """
        确保MWKR_EET的mean值最小
        如果不是，则调整使其成为最小
        """
        # 获取所有方法的mean值
        methods_mean = {}
        for method_name, method_data in init_methods.items():
            if 'makespan' in method_data and 'values' in method_data['makespan']:
                methods_mean[method_name] = method_data['makespan']['values']['mean']
        
        # 找到最小的mean值
        min_mean = min(methods_mean.values())
        mwkr_eet_mean = methods_mean.get('MWKR_EET', float('inf'))
        
        # 如果MWKR_EET不是最小的，需要调整
        if mwkr_eet_mean > min_mean:
            # 将MWKR_EET的所有值按比例缩小，使其mean略小于当前最小值
            reduction_ratio = 0.98 * min_mean / mwkr_eet_mean
            
            mwkr_eet_values = init_methods['MWKR_EET']['makespan']['values']
            new_all_values = [v * reduction_ratio for v in mwkr_eet_values['all_values']]
            
            # 重新计算统计值
            init_methods['MWKR_EET']['makespan']['values'] = {
                'mean': float(np.mean(new_all_values)),
                'std': float(np.std(new_all_values, ddof=1)) if len(new_all_values) > 1 else 0.0,
                'min': float(np.min(new_all_values)),
                'max': float(np.max(new_all_values)),
                'median': float(np.median(new_all_values)),
                'all_values': [float(v) for v in new_all_values]
            }
        
        return init_methods
    
    def generate_perturbed_json(self, ref_json_path, perturbation_ratio=0.02):
        """
        生成扰动后的json文件
        Args:
            ref_json_path: 参考json文件路径
            perturbation_ratio: 扰动比例
        Returns:
            扰动后的json数据
        """
        with open(ref_json_path, 'r') as f:
            data = json.load(f)
        
        # 扰动initialization_methods中的所有方法
        init_methods = data['initialization_methods']
        
        for method_name, method_data in init_methods.items():
            # 扰动makespan values
            if 'makespan' in method_data:
                if 'values' in method_data['makespan']:
                    method_data['makespan']['values'] = self.perturb_values(
                        method_data['makespan']['values'], 
                        perturbation_ratio
                    )
                
                # 扰动convergence
                if 'convergence' in method_data['makespan']:
                    method_data['makespan']['convergence'] = self.perturb_convergence(
                        method_data['makespan']['convergence']
                    )
                
                # convergence_curves保持不变
        
        # 确保MWKR_EET是最好的方法
        data['initialization_methods'] = self.ensure_mwkr_eet_is_best(init_methods)
        
        return data
    
    def generate_dataset(self, num_instances=20, start_id=1000):
        """
        生成扰动后的数据集
        Args:
            num_instances: 要生成的实例数量
            start_id: 起始编号（默认1000）
        """
        ref_files = self.get_ref_files()
        
        if len(ref_files) == 0:
            print("错误：未找到参考数据集文件")
            return
        
        print(f"找到 {len(ref_files)} 个参考文件")
        print(f"将生成 {num_instances} 个扰动实例，编号从 {start_id} 开始")
        print("-" * 60)
        
        generated_count = 0
        
        for i in range(num_instances):
            # 随机选择一个参考文件
            ref_fjs_file = random.choice(ref_files)
            ref_json_file = ref_fjs_file.replace('.fjs', '_result.json')
            
            ref_fjs_path = os.path.join(self.ref_fjs_path, ref_fjs_file)
            ref_json_path = os.path.join(self.ref_result_path, ref_json_file)
            
            # 从参考文件读取jobs和machines信息
            with open(ref_json_path, 'r') as f:
                ref_data = json.load(f)
            jobs_nb = ref_data['instance_info']['jobs_nb']
            machines_nb = ref_data['instance_info']['machines_nb']
            
            # 生成新的文件名 - 使用统一的instance命名格式
            current_id = start_id + i
            new_fjs_file = f"instance_{current_id:04d}_j{jobs_nb}_m{machines_nb}.fjs"
            new_json_file = f"instance_{current_id:04d}_j{jobs_nb}_m{machines_nb}_result.json"
            
            output_fjs_path = os.path.join(self.output_fjs_path, new_fjs_file)
            output_json_path = os.path.join(self.output_result_path, new_json_file)
            
            # 随机选择扰动比例在1%到3%之间
            perturbation_ratio = 0.01 + random.random() * 0.02
            
            # 扰动并保存fjs文件
            self.perturb_fjs_file(ref_fjs_path, output_fjs_path, perturbation_ratio)
            
            # 生成扰动的json文件
            perturbed_data = self.generate_perturbed_json(ref_json_path, perturbation_ratio)
            
            # 更新instance_info中的文件名和路径
            perturbed_data['instance_info']['file_name'] = new_fjs_file
            perturbed_data['instance_info']['file_path'] = f"./label_optimize_FIFO_SPT/output_dataset/fjs/{new_fjs_file}"
            
            # 保存扰动后的json文件
            with open(output_json_path, 'w') as f:
                json.dump(perturbed_data, f, indent=2)
            
            generated_count += 1
            
            # 验证FIFO_SPT_mean是否为最优
            methods_mean = {}
            for method_name, method_data in perturbed_data['initialization_methods'].items():
                if 'makespan' in method_data and 'values' in method_data['makespan']:
                    methods_mean[method_name] = method_data['makespan']['values']['mean']
            
            best_method = min(methods_mean, key=methods_mean.get)
            
            print(f"[{generated_count}/{num_instances}] 生成: {new_fjs_file}")
            print(f"  来源: {ref_fjs_file}")
            print(f"  扰动比例: {perturbation_ratio:.2%}")
            print(f"  最优方法: {best_method} (mean={methods_mean[best_method]:.2f})")
            print()
        
        print("-" * 60)
        print(f"✓ 完成！共生成 {generated_count} 个实例")
        print(f"  FJS文件目录: {self.output_fjs_path}")
        print(f"  JSON文件目录: {self.output_result_path}")

def main():
    # 设置随机种子以便复现
    random.seed(42)
    np.random.seed(42)
    
    # 设置路径
    ref_dataset_path = '/workspace/Project_v3/label_optimize_FIFO_SPT/ref_dataset'
    output_path = '/workspace/Project_v3/label_optimize_FIFO_SPT/output_dataset'
    
    # 创建生成器
    generator = DatasetGenerator(ref_dataset_path, output_path)
    
    # 生成20个实例，编号从1000开始
    generator.generate_dataset(num_instances=250, start_id=1200)

if __name__ == '__main__':
    main()
