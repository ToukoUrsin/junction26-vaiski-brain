"""On a sim box: make the sim spawn triangular prisms LYING on a rectangular face (ridge up), as in the organisers'
real frames, when ~/LYING exists. Test only; backup in layout.py.orig."""
import os, shutil
p = os.path.expanduser("~/aalto/aaltonomous/sim/aalto_task/layout.py")
s = open(p).read()
if "LYING" not in s:
    shutil.copy(p, p + ".orig")
    a = '        poses[pick] = (p, yaw_quat_xyzw(mat_yaw_to_pelvis(ob.get("yaw", 0.0))))\n'
    b = '''        q = yaw_quat_xyzw(mat_yaw_to_pelvis(ob.get("yaw", 0.0)))
        import os as _os
        if shape == "triangle" and _os.path.exists(_os.path.expanduser("~/LYING")):
            th = mat_yaw_to_pelvis(ob.get("yaw", 0.0))
            s_, c_, r_ = np.sin(th / 2), np.cos(th / 2), np.sqrt(0.5)
            q = np.array([c_ * r_, s_ * r_, s_ * r_, c_ * r_])          # yaw about z after +90 deg about mesh x: apex up
            p[2] += sc.size / 3 - sc.object_height("triangle") / 2      # centroid height on a rectangular face
        poses[pick] = (p, q)
'''
    assert s.count(a) == 1
    open(p, "w").write(s.replace(a, b))
    print("patched")
else:
    print("already patched")
