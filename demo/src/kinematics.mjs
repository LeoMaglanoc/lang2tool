import * as T from 'three';
// URDF origins are fixed-axis roll/pitch/yaw: Rz * Ry * Rx.
export function origin(object, xyz, rpy) {
  object.position.fromArray(xyz);
  object.quaternion.setFromEuler(new T.Euler(...rpy, 'ZYX'));
}
// Keep the joint origin separate from its rotation about the local joint axis.
export function connectJoint(parent, child, joint) {
  const fixed = new T.Group(); origin(fixed, joint.xyz, joint.rpy); parent.add(fixed);
  const moving = new T.Group(); fixed.add(moving); moving.add(child);
  return moving;
}
export function jointPose(node, joint, value) {
  const axis = node.userData.jointAxis ??= new T.Vector3(...joint.axis).normalize();
  if (joint.type === 'prismatic') node.position.copy(axis).multiplyScalar(value);
  else node.quaternion.setFromAxisAngle(axis, value);
}
