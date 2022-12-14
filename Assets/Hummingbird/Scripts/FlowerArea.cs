using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

/// <summary>
/// Manages a collection of flower plants and attached flowers in addition
/// to the hummingbirds and hunter drone agents that are in the area
///
/// This class acts as the controller in the environment
/// </summary>
public class FlowerArea : MonoBehaviour
{
    // The diameter of the area where the agent and flower 
    // can be used for observing relative distances from the agent to the flower
    public const float AreaDiameter = 20f;

    // The list of all flower plants in this area(flower plants have multiple flowers)
    private List<GameObject> flowerPlants;

    // A lookup dictionary for looking up a flower from a nectar collider
    private Dictionary<Collider, Flower> nectarFlowerDictionary;

    // The hashtable containing a count of collisions with each humming bird
    public Hashtable hummingBirdCollision;

    // The multi agent group for the hummingbirds
    public SimpleMultiAgentGroup mAgentGroupBird;
    public SimpleMultiAgentGroup mAgentGroupHunter;

    public int eliminatedCount;

    /// <summary>
    /// The list of all flowers in the area
    /// </summary>  
    public List<Flower> Flowers
    {
        get;
        private set;
    }

    /// <summary>
    /// The list of all HummingBirds in the area
    /// </summary>  
    public List<HummingBirdAgent> HummingBirds
    {
        get;
        private set;
    }

    /// <summary>
    /// The list of all hunters in the area
    /// </summary>
    public List<HunterAgent> Hunters
    {
        get;
        private set;
    }

    /// <summary>
    /// Max Academy steps before this platform resets
    /// </summary>
    /// <returns></returns>
    [Header("Max Environment Steps")] public int MaxEnvironmentSteps = 25000;
    private int m_ResetTimer;

    // ========================================================================

    /// <summary>
    /// Called when the game starts
    /// </summary> 
    private void Start()
    {
        mAgentGroupBird = new SimpleMultiAgentGroup();
        mAgentGroupHunter = new SimpleMultiAgentGroup();
        // Finds all flower, bird, and hunter children of the floating island transform
        FindChildFlowers(transform);
        FindChildBirds(transform);
        FindChildHunters(transform);
        eliminatedCount = 0;

        // register all birds within the group
        foreach(HummingBirdAgent bird in HummingBirds)
        {
            mAgentGroupBird.RegisterAgent(bird);
        }

        // register all hunters within the group 
        foreach(HunterAgent hunter in Hunters)
        {
            mAgentGroupHunter.RegisterAgent(hunter);
        }
        ResetScene();
    }

    /// <summary>
    /// Called when the area wakes up
    /// </summary> 
    private void Awake()
    {
        // Initialize variables 
        flowerPlants = new List<GameObject>();
        nectarFlowerDictionary = new Dictionary<Collider, Flower>();
        Flowers = new List<Flower>();
        HummingBirds = new List<HummingBirdAgent>();
        Hunters = new List<HunterAgent>();
        hummingBirdCollision = new Hashtable();
    }

    /// <summary>
    /// Reset the agent when an episode begins
    /// </summary> 
    public void ResetScene()
    {
        // Reset the flower birds and hunters in the area
        ResetFlowers();
        ResetHummingBirds();
        ResetHunters();
        eliminatedCount = 0;
        hummingBirdCollision.Clear();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        m_ResetTimer += 1;

        if (m_ResetTimer == MaxEnvironmentSteps || eliminatedCount == HummingBirds.Count)
        {
            EndGame();
        }

        if (m_ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            mAgentGroupBird.GroupEpisodeInterrupted();
            mAgentGroupHunter.GroupEpisodeInterrupted();
            ResetScene();
            m_ResetTimer = 0;
        }

    }

    public void EndGame()
    {
        Debug.Log("Reset with " + eliminatedCount + " birds eliminated");

        // if the hunter has eliminated more than 3/4 of birds then hunter wins
        if (eliminatedCount > (HummingBirds.Count * 0.75))
        {
            /// the quicker the episode, the better the reward
            Debug.Log("Hunters awarded with : " + (1 - (float)m_ResetTimer / MaxEnvironmentSteps));
            Debug.Log("Birds awarded with : " +  -1);
            mAgentGroupHunter.AddGroupReward(1 - ((float)m_ResetTimer / MaxEnvironmentSteps));
            mAgentGroupBird.AddGroupReward(-1);
        }
        // if more than 1/4 of the birds survive
        else
        {
            Debug.Log("Birds awarded with : " + 1);
            Debug.Log("Hunters awarded with : " + -1);

            mAgentGroupHunter.AddGroupReward(1);
            mAgentGroupBird.AddGroupReward(0.5f);
        }
        // end group episode for agents
        mAgentGroupBird.EndGroupEpisode();
        mAgentGroupHunter.EndGroupEpisode();
        ResetScene();
    }
    // ========================================================================

    /// <summary>
    /// Reset the flowers and flower plants in the area
    /// </summary> 
    public void ResetFlowers(){
        // Rotate each flower plant around the Y axis and subtly around the X and Z
        foreach (GameObject flowerPlant in flowerPlants)
        {
            float xRotation = Random.Range(-5f, 5f);
            float yRotation = Random.Range(-180f, 180f);
            float zRotation = Random.Range(-5f, 5f);
            flowerPlant.transform.localRotation = Quaternion.Euler(xRotation, yRotation, zRotation);
        }
        // Reset each flower
        foreach(Flower flower in Flowers)
        {
            flower.ResetFlower();
        }
    }

    /// <summary>
    /// Reset the humming birds in the area
    /// </summary>
    public void ResetHummingBirds()
    {
        foreach (HummingBirdAgent bird in HummingBirds)
        {
            bird.ResetBird();
        }
    }

    /// <summary>
    /// Reset the hunters in the area
    /// </summary>
    public void ResetHunters()
    {
        foreach (HunterAgent hunter in Hunters)
        {
            hunter.ResetHunter();
        }
    }

    /// <summary>
    /// Gets the <see cref="Flower"/> that the nectar belongs to
    /// <param name="collider">The Nectar Collider</param>
    /// <return>The matching flower</return>
    /// </summary> 
    public Flower GetFlowerFromNectar(Collider collider)
    {
        return nectarFlowerDictionary[collider];
    }

    /// <summary>
    /// Recursively finds all flowers and flower plants that are children of a parent transform
    /// <param name="parent">The parent of the children to check</param>
    /// </summary> 
    private void FindChildFlowers(Transform parent)
    { 
        for(int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.CompareTag("flower_plant"))
            {
                // found a flower plant, add it to the flower plants list
                flowerPlants.Add(child.gameObject);
                
                // look for flowers within the flower plant
                FindChildFlowers(child);
            }
            else
            {
                // Not a flower plant, look for a flower component
                if (child.TryGetComponent<Flower>(out var flower))
                {
                    // found a flower, add it to teh flower's list
                    Flowers.Add(flower);

                    // add nectar colliders to the nectar/flower dictionary
                    nectarFlowerDictionary.Add(flower.nectarCollider, flower);
                    // note, there are no floweres that are children of other flowers
                }
                else
                {
                    // flower component not found, so check children
                    FindChildFlowers(child);
                }
            }
        }
    }


    /// <summary>
    /// Find all humming birds that are children of the parent
    /// <param name="parent">The parent of the children to check</param>
    /// </summary>
    private void FindChildBirds(Transform parent)
    {
        for(int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.CompareTag("humming_bird") && !HummingBirds.Contains(child.GetComponent<HummingBirdAgent>()))
            {
                HummingBirds.Add(child.GetComponent<HummingBirdAgent>());
            }
        }
    }
    /// <summary>
    /// Find all hunters that are children of the parent
    /// </summary>
    /// <param name="parent">The parent of the chilren to check</param>
    private void FindChildHunters(Transform parent)
    {
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.CompareTag("hunter_agent") && !Hunters.Contains(child.GetComponent<HunterAgent>()))
            {
                Hunters.Add(child.GetComponent<HunterAgent>());
            }
        }
    }
}
