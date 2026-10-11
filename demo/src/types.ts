export type Pose = number[];
export interface Tool { id: string; name: string; family: string; asset: string }
export interface Example { id: string; tool: string; method: string; task: string; target: number[]; completion: number; success: boolean; duration: number; frames: number; data: string; translationRmse: number; rotationRmse: number; provenance: {run: string; geometryTrial: string; executionTrial: string; source: string; sha256: string; goalTransform: string} }
export interface Manifest { tools: Tool[]; examples: Example[]; robotPosition: number[]; tableTop: number }
export interface Recording { times: number[]; tool: Pose[]; goal: Pose[]; joints: number[][]; jointNames: string[]; reference: Pose[]; referenceInterval: number; referenceDuration: number; goalIndices: number[]; events: (string|null)[] }
export interface Visual { xyz: number[]; rpy: number[]; color: number[]; mesh?: string; scale?: number[]; box?: number[]; cylinder?: number[]; sphere?: number }
export interface Joint { name: string; type: string; parent: string; child: string; xyz: number[]; rpy: number[]; axis: number[]; mimic: {joint: string; multiplier: number; offset: number}|null }
export interface Graph { links: {name: string; visuals: Visual[]}[]; joints: Joint[] }
export interface MeshInfo { vertices: number; indices: number; file: string }
