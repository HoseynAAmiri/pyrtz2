import os
from pyrtz2 import afm

path = "C:/Users/hamiri7/Desktop/Hoseyn/AFM/Tong/240322 JASP-treated TCells day 1 50nM/"
exp_name = "con111"
labels = ['jas', 'con', 'cell', 'm']
probe_diameter = 5.4e-6
indentation = 1.0e-6

exp1 = afm.AFM(path, exp_name, labels, probe_diameter)

# key = exp1.curve_keys[20]
# print(key)
# curve = exp1.experiment[key]

# detected_contact_index = curve.detect_contact()
# curve.set_contact_index(detected_contact_index)
# curve.adjust_to_contact()
# fig = curve.get_dwell_relaxation_fig()
# fig = curve.add_dwell_relaxation_fit()


# fig.show()
exp1.experiment.update_annotations_from_file(
    os.path.join(path, exp_name) + "_cp_annotations.json")
exp1.experiment.update_annotations_from_file(
    os.path.join(path, exp_name) + "_vd_annotations.json")
exp1.experiment.set_cp_by_annotations()
exp1.experiment.set_vd_by_annotations()
results = exp1.experiment.get_fit_all(probe_diameter, indentation)
# fig = curve.get_dwell_relaxation_fig()
# print(fig)
results.to_csv(path + "/" + exp_name + "_fit_results.csv", index=False)
merger = exp1.experiment.export_figures()
merger.write(path + "/" + exp_name + "_fit_results.pdf")
# key = exp1.curve_keys[3]
# print(key)
# curve = exp1.experiment[key]
# fig = curve.get_dwell_relaxation_fig()
# fig = curve.add_dwell_relaxation_fit()
# fig.show()
'''
REMOVE FIT:
removing = [i
            for i, trace in enumerate(fig['data']) if trace['name'] and trace['mode'] == 'lines']
data_list = list(fig['data'])
for i in removing[::-1]:
    data_list.pop(i)
fig['data'] = tuple(data_list)

curve.adjust_to_contact()
fit = curve.fit_relaxation()
print(fit[0], fit[1])
relaxation = curve.get_relaxation()
plt.plot(relaxation['t'], relaxation['ind'])
plt.plot(relaxation['t'].values[:len(fit[-1])], fit[-1])
plt.show()
'''
