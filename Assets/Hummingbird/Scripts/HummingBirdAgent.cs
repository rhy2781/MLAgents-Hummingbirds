using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using static HunterAgent;

/// <summary>
/// A humming bird machine learning agent
/// </summary> 
public class HummingBirdAgent : Agent
{
    [Tooltip("Force to apply when moving")]
    public float moveForce = 2f;

    [Tooltip("Speed to pitch up or down")]
    public float pitchSpeed = 100f;

    [Tooltip("Speed to rotate around the up axis")]
    public float yawSpeed = 100f;

    [Tooltip("Transform at the tip of the beak")]
    public Transform beakTip;

    [Tooltip("The agent's Camera")]
    public Camera agentCamera;

    [Tooltip("Whether this is training mode or Gameplay mode")]
    public bool trainingMode;

    // The rigid body of the agent
    private Rigidbody rigidBody;

    // The flower area that the agent is in 
    private FlowerArea flowerArea;

    // The nearest flower to the agent
    private Flower nearestFlower;

    // The next flower that the agent will fly to
    private Flower nextFlower;

    // Allows for smoother pitch changes
    private float smoothPitchChange = 0f;

    // Allows for smoother yaw changes
    private float smoothYawChange = 0f;

    // Maximum angle that the bird can pitch up or down
    private const float maxPitchAngle = 80f;

    // Maximum distance from the beak tip to accept nectar collision
    private const float beakTipRadius = 0.008f;

    // Whether the agent is frozen(intentionally not fying)
    // observing the environment, but not taking any actions
    private bool frozen = false;

    /// <summary>
    /// The amount of nectar this agent has obtained this episode
    /// </summary> x
    public float nectarObtained
    {
        get;
        private set;
    }

    /// <summary>
    /// Initialize the agent
    /// </summary> 
    public override void Initialize()
    {
        rigidBody = GetComponent<Rigidbody>();
        flowerArea = GetComponentInParent<FlowerArea>();

        // If not training mode, no max step, play forever
        if (!trainingMode) MaxStep = 0;
    }


    /// <summary>
    /// Reset the agent when an episode begins
    /// </summary> 
    public override void OnEpisodeBegin()
    {
        if (trainingMode)
        {
            // Only reset flowers in training when there is one agent per area
            flowerArea.ResetFlowers();
            //flowerArea.ResetHummingBirds();
            //flowerArea.ResetHunterAgents();
        }

        // Reset the nectar Obtained
        nectarObtained = 0;

        // Zero out velocities so that movement stops before a new episode begins
        rigidBody.velocity = Vector3.zero;
        rigidBody.angularVelocity = Vector3.zero;

        // Default to spawning in front of a flower
        bool inFrontOfFlower = true;

        // Move the agent to a new Position
        MoveToSafeRandomPosition(inFrontOfFlower);

        // Recalculate the newest flower now that the agent has moved
        UpdateNearestFlower();
    }

    /// <summary>
    /// Called when an action is received from either the player or the neural network
    ///
    /// vectorAction[i] represents:
    /// Index 0: move vector x (+1 right, -1 left)
    /// Index 1: move vector y (+1 up, -1 down)
    /// Index 2: move vector z (+1 forward, -1 backward)
    /// Index 3: pitch angle (+1 pitch up, -1 pitch down)
    /// Index 4: yaw angle (+1 turn right, -1 turn left)
    /// <paran name="actionBuffers">The actions to take</param>
    /// </summary> 

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        // Don't take actions if frozen
        if (frozen) return;

        var continuousActions = vectorAction.ContinuousActions;

        // calculate movement vector
        Vector3 move = new Vector3(continuousActions[0], continuousActions[1], continuousActions[2]);

        // add force in the direction of the move vector
        rigidBody.AddForce(move * moveForce);

        // Get the current rotation
        Vector3 rotationVector = transform.rotation.eulerAngles;

        // calculate pitch and yaw rotation 
        float pitchChange = continuousActions[3];
        float yawChange = continuousActions[4];

        // calculate mooth rotation changes
        smoothPitchChange = Mathf.MoveTowards(smoothPitchChange, pitchChange, 2f * Time.fixedDeltaTime);
        smoothYawChange = Mathf.MoveTowards(smoothYawChange, yawChange, 2f * Time.fixedDeltaTime);

        // calculate new pitch and yaw based on new smoothed values
        // clamp pitch to avoid flipping upside down
        float pitch = rotationVector.x + smoothPitchChange * Time.fixedDeltaTime * pitchSpeed;
        if (pitch > 180) pitch -= 360f;
        pitch = Mathf.Clamp(pitch, -maxPitchAngle, maxPitchAngle);

        float yaw = rotationVector.y + smoothYawChange * Time.fixedDeltaTime * yawSpeed;

        // apply the new rotation
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
    }

    /// <summary>
    /// Collect vector observations from the environment
    /// <param name="sensor">The vector sensor</param>
    /// </summary> 
    public override void CollectObservations(VectorSensor sensor)
    {
        // If nearest flower is null, observe an empty array and return early
        if (nearestFlower == null)
        {
            sensor.AddObservation(new float[10]);
            return;
        }

        // Observe the agent's local rotation (4 observations)
        sensor.AddObservation(transform.localRotation.normalized); // Quaternion with magitude of 1

        // Get a vector from the beak tip to the nearest flower
        Vector3 toFlower = nearestFlower.FlowerCenterPosition - beakTip.position;

        // Observe a normalized vector pointing to the nearest flower(3 observations)
        sensor.AddObservation(toFlower.normalized);

        // Observe a dot product that indicates whether the beak tip is in front of the flower(1 observation)
        // (+1 means that the beak tip is directly in front of the flower, -1 means directly behind)
        sensor.AddObservation(Vector3.Dot(toFlower.normalized, -nearestFlower.FlowerUpVector.normalized)); // beak to flower and down vector into the flower

        // Observe a dot product that indicates whether the beak is pointing towards the flower(1 observation)
        // (+1 means that the beak is pointing directly at the flower, -1 means directly away)
        sensor.AddObservation(Vector3.Dot(beakTip.forward.normalized, -nearestFlower.FlowerUpVector.normalized)); // orientation of the beak to the flower and down vector into the flower

        // Observe the relative distance from the beak tip to the flower(1 observation)
        sensor.AddObservation(toFlower.magnitude / FlowerArea.AreaDiameter); // ratio to the flower compared to total island diameter

        // 10 total observations
    }

    /// <summary>
    /// When behavior type is set to hueristic only on the agent's behavior parameters,
    /// this function will be called. It's return values will be fed into
    /// <see cref="OnActionReceived(float[])">Instead of using the neural network</cref>
    /// <param name="actionsOut">An output action array</param>
    /// </summary>
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;

        // Create placeholders for all movement/turning
        Vector3 forward = Vector3.zero;
        Vector3 left = Vector3.zero;
        Vector3 up = Vector3.zero;
        float pitch = 0f;
        float yaw = 0f;

        // Forward - Backward
        if (Input.GetKey(KeyCode.W)) forward = transform.forward;
        else if (Input.GetKey(KeyCode.S)) forward = -transform.forward;

        // Left - Right
        if (Input.GetKey(KeyCode.A)) left = -transform.right;
        else if (Input.GetKey(KeyCode.D)) left = transform.right;

        // Up - Down
        if (Input.GetKey(KeyCode.E)) up = transform.up;
        else if (Input.GetKey(KeyCode.C)) up = -transform.up;

        // Pitch up - Pitch down
        if (Input.GetKey(KeyCode.UpArrow)) pitch = -0.5f;
        else if (Input.GetKey(KeyCode.DownArrow)) pitch = 0.5f;

        // Turn left - Turn right
        if (Input.GetKey(KeyCode.LeftArrow)) yaw = -0.5f;
        else if (Input.GetKey(KeyCode.RightArrow)) yaw = 0.5f;

        // Combin the movement vectors and normalize
        Vector3 combined = (forward + left + up).normalized;

        continuousActions[0] = combined.x;
        continuousActions[1] = combined.y;
        continuousActions[2] = combined.z;
        continuousActions[3] = pitch;
        continuousActions[4] = yaw;
    }

    /// <summary>
    /// Pevent the agent from moving and taking actions
    /// </summary>
    public void FreezeAgent()
    {
        Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = true;
        rigidBody.Sleep();
    }

    /// <summary>
    /// Resume Agent movement and actions
    /// </summary>
    public void UnfreezeAgent()
    {
        Debug.Assert(trainingMode == false, "Freeze/Unfreeze not supported in training");
        frozen = false;
        rigidBody.WakeUp();
    }

    /// <summary>
    /// Move the agent into a safe random posotion i.e. does not collide anything 
    /// If also in front of flower, point the beak in front of the flower
    /// </summary> 
    private void MoveToSafeRandomPosition(bool inFrontOfFlower)
    {
        bool safePositionFound = false;
        int attemptsRemaining = 100; // Prevents infinite loop
        Vector3 potentialPosition = Vector3.zero;
        Quaternion potentialRotation = new Quaternion();

        // loop until a safe position is found, or we run out of attempts
        while(!safePositionFound && attemptsRemaining > 0)
        {
            attemptsRemaining --;
            if(inFrontOfFlower)
            {
                // Pick a random flower
                Flower randomFlower = flowerArea.Flowers[UnityEngine.Random.Range(0, flowerArea.Flowers.Count)];

                // Position 10-20 cm in from of the flower
                //float distanceFromFlower = UnityEngine.Random.Range(.1f, .2f); disabled
                potentialPosition = randomFlower.transform.position + randomFlower.FlowerUpVector * .1f;

                // Point beak at flower(bird's head is center of transform)
                Vector3 toFlower = randomFlower.FlowerCenterPosition - potentialPosition;
                potentialRotation = Quaternion.LookRotation(toFlower, Vector3.up); 
            }

            // Check to see if the agent will collide with anything
            Collider[] colliders = Physics.OverlapSphere(potentialPosition, 0.05f);
        
            // Sage position has been found is no colliders have overlapped
            safePositionFound = colliders.Length == 0;
        }
        Debug.Assert(safePositionFound, "Could not find a safe position to spawn");

        transform.position = potentialPosition;
        transform.rotation = potentialRotation;
    }

    /// <summary>
    /// Update the nearest flower to the agent
    /// </summary> 
    private void UpdateNearestFlower()
    {
        nextFlower = flowerArea.Flowers[0];
        foreach(Flower potentialFlower in flowerArea.Flowers)
        {
            if (potentialFlower.hasNectar)
            {
                float distanceToNextFlower = Vector3.Distance(nextFlower.transform.position, beakTip.position);
                float distanceToPotentialFlower = Vector3.Distance(potentialFlower.transform.position, beakTip.position);

                if(distanceToPotentialFlower < distanceToNextFlower)
                {
                    nextFlower = potentialFlower;
                }
            }
        }

        nearestFlower = nextFlower;

       
        foreach(Flower flower in flowerArea.Flowers)
        {
            if(nearestFlower == null && flower.hasNectar) // no nearest flower
            {
                // no current nearest flower and this flower has nectar
                nearestFlower = flower;
            }

            // Calculate distance to this flower and distance to the current nearest flower
            float distanceToFlower = Vector3.Distance(flower.transform.position, beakTip.position);
            float distanceToCurrentNearestFlower = Vector3.Distance(nearestFlower.transform.position, beakTip.position);

            // If current nearest flower is empty of this flower is closer, update nearest flower

            if (!nearestFlower.hasNectar || distanceToFlower < distanceToCurrentNearestFlower) // nearest flower has nectar
            {
                nearestFlower = flower;
            }
        }
    }

    /// <summary>
    /// Called when the agent's collider enters a trigger collider
    /// </summary>
    /// <param name="other">The trigger collider</param>
    private void OnTriggerEnter(Collider other)
    {
        TriggerEnterOrStay(other);
    }

    /// <summary>
    /// Called when the agent's collider stays in a trigger collider
    /// </summary>
    /// <param name="other">The trigger collider</param>
    private void OnTriggerStay(Collider other)
    {
        TriggerEnterOrStay(other);
    }

    /// <summary>
    /// Handles when the agent's collider enters or stays in a trigger collider
    /// </summary>
    /// <param name="collider">The trigger collider</param>
    private void TriggerEnterOrStay(Collider collider)
    {
        // Check if agent is colliding with nectar
        if(collider.CompareTag("nectar"))
        {
            Vector3 closestPointToBeakTip = collider.ClosestPoint(beakTip.position);


            // Check if the closest collision point is close to the beak tip
            // Note: A collision with anything but the beak tip should not count
            if (Vector3.Distance(beakTip.position, closestPointToBeakTip) < beakTipRadius)
            {
                // Look up the flower for this nectar collider
                Flower flower = flowerArea.GetFlowerFromNectar(collider);

                // Attempt to take .01 nectar
                // Note: this is per fixed timestep, meaning it happens every .02 seconds, or 50x per second
                float nectarReceived = flower.Feed(.01f);

                // Keep track of nectar obtained
                nectarObtained += nectarReceived;

                if(trainingMode)
                {
                    // Calculate the reward for getting nectar
                    float bonus = .2f * Mathf.Clamp01(Vector3.Dot(transform.forward.normalized, -nearestFlower.FlowerUpVector.normalized));
                    AddReward(.1f + bonus);

                    // If flower is empty, update nearest flower
                    if (!flower.hasNectar)
                    {
                        UpdateNearestFlower();
                    }
                }
            }
        }
    }

    /// <summary>
    /// Called when the agent collides with something solid
    /// </summary>
    /// <param name="collision">The collision info</param>
    private void OnCollisionEnter(Collision collision)
    {
        if(trainingMode && collision.collider.CompareTag("boundary"))
        {
            // Collided with the area boundary, give a negative reward
            AddReward(-.5f);
        }
        if(collision.gameObject.CompareTag("hunter_agent"))
        {
            // hunter attacked the humming bird
            AddReward(-.5f);
        }
    }

    /// <summary>
    /// Called every frame
    /// </summary>
    private void Update()
    {
        // Draw a line from the beak tip to the nearest flower
        if(nearestFlower != null)
        {
            Debug.DrawLine(beakTip.position, nearestFlower.FlowerCenterPosition, Color.black);
        }

    }

    /// <summary>
    /// Called every .02 seconds
    /// </summary>
    private void FixedUpdate()
    {
        if (nearestFlower != null && !nearestFlower.hasNectar)
        {
            // avoids nearest scenario where nearest flower nectar is stolen by opponent and not updated
            UpdateNearestFlower();
        }
    }
    public void ResetBird()
    {
        gameObject.SetActive(true);
    }
}
