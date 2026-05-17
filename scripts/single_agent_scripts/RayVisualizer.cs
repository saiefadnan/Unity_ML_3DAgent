using UnityEngine;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

public class RayVisualizer : MonoBehaviour
{
    public RayPerceptionSensorComponent3D sensor;

    // Each ray result cached every Update, drawn every OnRenderObject
    private struct RayResult
    {
        public Vector3 start;
        public Vector3 end;
        public Color color;
    }

    private List<RayResult> rayResults = new List<RayResult>();
    private static Material lineMat;

    void EnsureMaterial()
    {
        if (lineMat == null)
        {
            // Hidden/Internal-Colored is always available in Unity
            lineMat = new Material(Shader.Find("Hidden/Internal-Colored"));
            lineMat.hideFlags = HideFlags.HideAndDontSave;
            lineMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            lineMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            lineMat.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            lineMat.SetInt("_ZWrite", 0);
        }
    }

    void Update()
    {
        if (sensor == null) return;
        CastRays();
    }

    void CastRays()
    {
        rayResults.Clear();

        float rayLength = sensor.RayLength;
        int raysPerSide = sensor.RaysPerDirection;
        float maxAngle = sensor.MaxRayDegrees;
        LayerMask mask = sensor.RayLayerMask;
        int totalRays = raysPerSide * 2 + 1;
        Transform origin = sensor.transform;

        for (int i = 0; i < totalRays; i++)
        {
            float angle = (totalRays > 1)
                ? Mathf.Lerp(-maxAngle, maxAngle, (float)i / (totalRays - 1))
                : 0f;

            // Yaw spread around sensor's UP axis — matches RayPerceptionSensorComponent3D
            Vector3 dir = Quaternion.AngleAxis(angle, origin.up) * origin.forward;

            RaycastHit hit;
            bool didHit = Physics.Raycast(origin.position, dir, out hit, rayLength, mask);

            Color col = Color.red; // no hit
            if (didHit)
            {
                if (hit.collider.CompareTag("Victim")) col = Color.green;
                else if (hit.collider.CompareTag("Ground")) col = Color.yellow;
                else if (hit.collider.CompareTag("Obstacle") || hit.collider.CompareTag("Wall")) col = Color.blue;
                else col = Color.cyan;
            }

            rayResults.Add(new RayResult
            {
                start = origin.position,
                end = didHit ? hit.point : origin.position + dir * rayLength,
                color = col
            });
        }
    }

    // OnRenderObject is called for every camera that renders the scene,
    // including the Game view, with no dependency on Gizmos being enabled.
    void OnRenderObject()
    {
        if (rayResults == null || rayResults.Count == 0) return;

        EnsureMaterial();
        lineMat.SetPass(0);

        GL.PushMatrix();
        GL.MultMatrix(Matrix4x4.identity); // rays are already in world space

        GL.Begin(GL.LINES);
        foreach (var r in rayResults)
        {
            GL.Color(r.color);
            GL.Vertex(r.start);
            GL.Vertex(r.end);
        }
        GL.End();

        GL.PopMatrix();
    }
}
