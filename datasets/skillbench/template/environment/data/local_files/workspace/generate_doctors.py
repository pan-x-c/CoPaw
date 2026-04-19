import json
import random
from datetime import datetime, timedelta

# 生成医生数据
departments = ["内科", "外科", "儿科", "妇科", "口腔科", "眼科", "皮肤科"]
doctors = []

for i, dept in enumerate(departments):
    num_doctors = random.randint(2, 4)
    for j in range(num_doctors):
        doctor = {
            "id": f"DOC{str(i+1).zfill(3)}{str(j+1).zfill(2)}",
            "name": f"张{'伟明华强军杰磊辉'[(i+j)%8]}医生",
            "department": dept,
            "title": random.choice(["主治医师", "副主任医师", "主任医师"]),
            "specialty": f"{dept}专业，擅长{random.choice(['常见病诊疗', '慢性病管理', '微创手术', '综合治疗'])}",
            "available_dates": [(datetime.now() + timedelta(days=k)).strftime("%Y-%m-%d") for k in range(1, 15)],
            "slots_per_day": random.randint(10, 30)
        }
        doctors.append(doctor)

with open("doctors_data.json", "w", encoding="utf-8") as f:
    json.dump({"doctors": doctors, "generated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

print(f"Generated {len(doctors)} doctors")
